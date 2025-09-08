/**
 * @file nvivafilter_imagecorrection.cpp
 * @brief GStreamer nvivafilter plugin for full GPU image correction.
 *
 * This is the main entrypoint for the plugin loaded by GStreamer pipelines.
 *
 * Responsibilities:
 *  - Zero-copy map EGLImage → CUDA device pointers (NV12 planes).
 *  - Stage 1: Fisheye rectification (optional).
 *  - Stage 2: Temporal denoise (reduces flicker/noise using previous frame).
 *  - Stage 3: Auto statistics (histogram + ROI + mean chroma).
 *  - Stage 4: Auto-exposure & gamma with anti-flicker smoothing.
 *  - Stage 5: Auto white balance (gray-world).
 *  - Stage 6: LUT-based color grading in-place (contrast/brightness/sat).
 *  - Stage 7: Local tone mapping (CLAHE-lite).
 *
 * Config:
 *  - JSON live reconfig (via RuntimeControls).
 *  - Env vars: ICP_STAGE (rectify/color/both), ICP_CONTROLS (path to JSON).
 *
 * Output:
 *  - Corrected NV12 frame written in-place → fed directly to downstream encoder.
 */
 
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <algorithm>
#include <cstdint>

#include <cuda.h>
#include <cudaEGL.h>

#include "nvivafilter_customer_api.hpp"
#include "rectify_config.hpp"
#include "kernel_rectify.cuh"
#include "kernel_color.cuh"
#include "runtime_controls.hpp"

// ============================================================================
// Stato globale
// ============================================================================
static CUcontext g_ctx = nullptr;

// Scratch per rettifica
static CUdeviceptr g_sY=0, g_sUV=0; static size_t g_pY=0, g_pUV=0; static int gW=0, gH=0;

// Prev frame per denoise
static CUdeviceptr g_pvY=0, g_pvUV=0; static size_t g_ppY=0, g_ppUV=0;

// Stage flags
static bool STAGE_RECTIFY = true;
static bool STAGE_COLOR   = true;

// Config
static icp::RectifyConfig g_cfg;
static RuntimeControls*   g_rc  = nullptr;

// AE/AGC stato
static struct {
    bool  inited = false;
    float k_exp  = 1.0f;
    float gamma  = 1.0f;
    float wb_r   = 1.0f;
    float wb_b   = 1.0f;
} g_auto;

static inline float clampf(float v, float a, float b){ return v<a?a:(v>b?b:v); }
static inline int   i_min(int a,int b){ return a<b?a:b; }
static inline int   i_max(int a,int b){ return a>b?a:b; }

static void ensure_cuda_ctx(){
    static bool inited=false; if(!inited){ cuInit(0); inited=true; }
    CUcontext cur=nullptr; cuCtxGetCurrent(&cur);
    if(!cur){ CUdevice dev=0; cuDeviceGet(&dev,0); cuCtxCreate(&g_ctx,0,dev); cuCtxSetCurrent(g_ctx); }
}
static void load_stage_flags_once(){
    static bool done=false; if(done) return; done=true;
    if(const char* s=getenv("ICP_STAGE")){
        std::string v(s); std::transform(v.begin(),v.end(),v.begin(),[](unsigned char c){return std::tolower(c);});
        if(v=="rectify"){STAGE_RECTIFY=true; STAGE_COLOR=false;}
        if(v=="color"){STAGE_RECTIFY=false; STAGE_COLOR=true;}
        if(v=="both"){STAGE_RECTIFY=true; STAGE_COLOR=true;}
    }
}
static void ensure_scratch(int W,int H){
    if(g_sY && g_sUV && gW==W && gH==H) return;
    if(g_sY){cuMemFree(g_sY); g_sY=0;} if(g_sUV){cuMemFree(g_sUV); g_sUV=0;}
    cuMemAllocPitch(&g_sY, &g_pY,  (size_t)W, (size_t)H,   4);
    cuMemAllocPitch(&g_sUV,&g_pUV, (size_t)W, (size_t)(H/2),4);
    gW=W; gH=H; fprintf(stderr,"[ic] scratch allocated %dx%d pitchY=%zu pitchUV=%zu\n",W,H,g_pY,g_pUV);
}
static void ensure_prev(int W,int H){
    if(g_pvY && g_pvUV && gW==W && gH==H) return;
    if(g_pvY){cuMemFree(g_pvY); g_pvY=0;} if(g_pvUV){cuMemFree(g_pvUV); g_pvUV=0;}
    cuMemAllocPitch(&g_pvY,  &g_ppY,  (size_t)W, (size_t)H,   4);
    cuMemAllocPitch(&g_pvUV, &g_ppUV, (size_t)W, (size_t)(H/2),4);
    // init: Y=0, UV=128
    cuMemsetD8(g_pvY,  0,   g_ppY  * (size_t)H);
    cuMemsetD8(g_pvUV, 128, g_ppUV * (size_t)(H/2));
}

static void copy_to_scratch(const uint8_t* dY,const uint8_t* dUV,size_t pY,size_t pUV,int W,int H){
    CUDA_MEMCPY2D c{}; c.srcMemoryType=CU_MEMORYTYPE_DEVICE; c.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    c.srcDevice=(CUdeviceptr)dY; c.srcPitch=pY; c.dstDevice=g_sY; c.dstPitch=g_pY; c.WidthInBytes=(size_t)W; c.Height=(size_t)H; cuMemcpy2D(&c);
    CUDA_MEMCPY2D c2{}; c2.srcMemoryType=CU_MEMORYTYPE_DEVICE; c2.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    c2.srcDevice=(CUdeviceptr)dUV; c2.srcPitch=pUV; c2.dstDevice=g_sUV; c2.dstPitch=g_pUV; c2.WidthInBytes=(size_t)W; c2.Height=(size_t)(H/2); cuMemcpy2D(&c2);
}

// nvivafilter stubs
static void pre_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}
static void post_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}

static void gpu_process(EGLImageKHR image, void **)
{
    ensure_cuda_ctx();
    load_stage_flags_once();

    CUgraphicsResource res=nullptr;
    if(cuGraphicsEGLRegisterImage(&res,image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)!=CUDA_SUCCESS){ fprintf(stderr,"[ic] cuGraphicsEGLRegisterImage failed\n"); return; }
    CUeglFrame f{};
    if(cuGraphicsResourceGetMappedEglFrame(&f,res,0,0)!=CUDA_SUCCESS){ fprintf(stderr,"[ic] GetMappedEglFrame failed\n"); cuGraphicsUnregisterResource(res); return; }
    if(f.frameType!=CU_EGL_FRAME_TYPE_PITCH || f.planeCount<2){ fprintf(stderr,"[ic] Unexpected frameType=%d planeCount=%d\n",f.frameType,f.planeCount); cuGraphicsUnregisterResource(res); return; }

    uint8_t* dY  = static_cast<uint8_t*>(f.frame.pPitch[0]);
    uint8_t* dUV = static_cast<uint8_t*>(f.frame.pPitch[1]);
    const int W = (int)f.width, H=(int)f.height, pitch=(int)f.pitch;

    icp::RectifyConfig cfg = g_rc ? g_rc->get() : g_cfg;

    // 1) Rettifica (GPU→GPU)
    if (STAGE_RECTIFY) {
        ensure_scratch(W,H);
        copy_to_scratch(dY,dUV,pitch,pitch,W,H);

        const float FOV_fish = cfg.fish_fov_deg * (float)M_PI / 180.f;
        const float f_fish   = cfg.r_f / (FOV_fish * 0.5f);
        const float fx       = (W * 0.5f) / std::tan(cfg.out_hfov_deg * (float)M_PI / 360.f);
        const float cx_rect  = W * 0.5f;
        const float cy_rect  = H * 0.5f;

        icp::launch_rectify_nv12(
            (const uint8_t*)(uintptr_t)g_sY,  W,H,(int)g_pY,
            (const uint8_t*)(uintptr_t)g_sUV,      (int)g_pUV,
            dY, W,H,pitch,
            dUV,     pitch,
            cfg.cx_f, cfg.cy_f, cfg.r_f,
            f_fish, fx, cx_rect, cy_rect,
            0);
    }

    // 2) Temporal denoise (prima delle stats/AE)
    ensure_prev(W,H);
    icp::launch_temporal_denoise_nv12(
        dY, dUV, pitch, pitch, W, H,
        (uint8_t*)(uintptr_t)g_pvY, (uint8_t*)(uintptr_t)g_pvUV,
        (int)g_ppY, (int)g_ppUV,
        /*alphaY*/0.35f, /*alphaUV*/0.25f,
        /*thrY*/6, /*thrUV*/8,
        0);

    // 3) Stats (ROI centrale)
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
                            hist, &mU, &mV, 0);

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

    // 4) AE anti-flicker + highlight-protect
    const float HI_CAP_Y  = 230.0f;
    const float HI_WEIGHT = 0.85f;
    const float CLIP_MAX  = 0.015f;

    float k_mid = clampf(cfg.target_Y / std::max(1, p50), 0.55f, 1.45f);
    float k_hi  = (p995 > 0) ? (HI_CAP_Y / (float)p995) : 1.0f;
    if (clipRatio > CLIP_MAX) k_hi = std::min(k_hi, 0.90f);

    float k_target = std::min(k_mid, std::pow(k_hi, HI_WEIGHT));

    // anti-flicker: step più lento quando si illumina, più veloce quando si scurisce
    const float STEP_UP   = std::max(0.02f, std::min(0.08f, cfg.auto_ae_step*0.7f));
    const float STEP_DOWN = std::max(0.05f, std::min(0.12f, cfg.auto_ae_step*1.3f));

    if (!g_auto.inited) g_auto.k_exp = k_target;
    else {
        float ratio = k_target / g_auto.k_exp;
        float step  = (ratio>1.f) ? STEP_UP : STEP_DOWN;
        ratio = clampf(ratio, 1.f - step, 1.f + step);
        g_auto.k_exp *= ratio;
    }

    float gamma_tgt = 1.0f;
    if (p50 < 90)                               gamma_tgt = cfg.auto_gamma_min;
    else if (p999 > 245 || clipRatio > CLIP_MAX) gamma_tgt = cfg.auto_gamma_max;
    if (!g_auto.inited) g_auto.gamma = gamma_tgt;
    else                g_auto.gamma = 0.95f*g_auto.gamma + 0.05f*gamma_tgt;
    float gammaFinal = cfg.gamma * g_auto.gamma;

    // 5) Filmic LUT (white al 99.9°)
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
        float n = (i/255.f) * g_auto.k_exp;
        n = n<0.f?0.f:(n>3.0f?3.0f:n);
        float y = filmicU2(n) / fw;
        y = y<0.f?0.f:(y>1.f?1.f:y);
        lut[i] = (uint8_t)std::lround(y * 255.f);
    }

    // 6) AWB
    float wb_r = cfg.wb_r, wb_g = cfg.wb_g, wb_b = cfg.wb_b;
    if (cfg.auto_wb) {
        float u = mU - 128.f, v = mV - 128.f, Ymid=110.f;
        float Rm = Ymid + 1.5748f*v;
        float Gm = Ymid - 0.1873f*u - 0.4681f*v;
        float Bm = Ymid + 1.8556f*u;
        float wr  = clampf((Gm+1e-3f)/(Rm+1e-3f), 1.f - cfg.auto_wb_clamp, 1.f + cfg.auto_wb_clamp);
        float wbb = clampf((Gm+1e-3f)/(Bm+1e-3f), 1.f - cfg.auto_wb_clamp, 1.f + cfg.auto_wb_clamp);
        if (!g_auto.inited) { g_auto.wb_r=wr; g_auto.wb_b=wbb; }
        else { g_auto.wb_r = 0.95f*g_auto.wb_r + 0.05f*wr; g_auto.wb_b = 0.95f*g_auto.wb_b + 0.05f*wbb; }
        wb_r *= g_auto.wb_r; wb_b *= g_auto.wb_b;
    }

    // 7) Grading globale (in-place)
    icp::launch_color_grade_nv12_inplace(
        dY, dUV, pitch, pitch, W, H,
        lut,
        cfg.contrast, cfg.brightness,
        cfg.saturation, gammaFinal,
        wb_r, wb_g, wb_b,
        /*sat rolloff*/0.70f, 0.97f, 0.25f,
        0);

    // 8) Local tone-mapping “CLAHE-lite” (9x9) su Y
    icp::launch_local_tonemap_nv12(
        dY, pitch, W, H,
        /*radius*/4,
        /*amount*/0.55f,
        /*hi_start*/0.70f, /*hi_end*/0.97f,
        0);

    g_auto.inited = true;

    cuCtxSynchronize();
    cuGraphicsUnregisterResource(res);
}

// init/deinit
extern "C" void init(CustomerFunction* f)
{
    fprintf(stderr,"[ic] init() loaded (rectify + color + denoise + LTM)\n");
    if(!f) return;

    if(const char* p = getenv("ICP_CONTROLS")){
        try { g_rc = new RuntimeControls(std::string(p), true);
              fprintf(stderr,"[ic] RuntimeControls watching: %s\n", p);
        } catch (...) { g_rc=nullptr; fprintf(stderr,"[ic] RuntimeControls disabled\n"); }
    } else {
        fprintf(stderr,"[ic] RuntimeControls disabled (set ICP_CONTROLS to enable)\n");
    }

    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;
}

extern "C" void deinit(void)
{
    if(g_sY){cuMemFree(g_sY); g_sY=0;} if(g_sUV){cuMemFree(g_sUV); g_sUV=0;}
    if(g_pvY){cuMemFree(g_pvY); g_pvY=0;} if(g_pvUV){cuMemFree(g_pvUV); g_pvUV=0;}
    if(g_rc){delete g_rc; g_rc=nullptr;}
    fprintf(stderr,"[ic] deinit()\n");
}
