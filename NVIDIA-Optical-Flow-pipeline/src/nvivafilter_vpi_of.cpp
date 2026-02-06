// nvivafilter_vpi_of.cpp
//
// Jetson Orin / JetPack 5.4.1 / VPI 2.4.x
//
// NVDEC -> NVMM(NV12) -> nvivafilter (this .so) -> NVMM
//
// VPI 2.4 + OFA Dense Optical Flow.
// OVERLAY: draws motion-vector arrows onto the NV12 luma plane (Y) via CUDA EGL interop.
//
// Current stability fix:
// - DO NOT wrap EGLImage into VPI on nvbuf-mem-surface-array pipelines.
// - Some pipelines keep the NVMM container locked exclusively during the callback.
// - Use CUDA-EGL to copy NV12 Y plane into a CUDA-backed VPI image, then operate only on VPI-owned buffers.
//
// Debug additions:
// - Optional MV stats + top-K spike print from mv_cpu
//   Enable with env: VPI_OF_MV_PRINT=1
//   Control cadence: VPI_OF_MV_PRINT_EVERY_N (default: VPI_OF_PRINT_EVERY_N)
//   Control topK:   VPI_OF_MV_PRINT_TOPK (default: 8)
//
// Robust overlay fix:
// - Detect "bad MV frames" (scene-cut / glitch) when typical motion explodes.
// - On bad frames: skip overlay + reset prev buffer to recover immediately.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cmath>

#include <EGL/egl.h>

#include <cuda.h>

#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Pyramid.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/OpticalFlowDense.h>

#include "nvivafilter_customer_api.hpp"
#include "overlay.hpp"
#include "egl_copy.hpp"   // egl_nv12_copy_y_to_cuda(...)

#define CHECK_VPI(stmt)                                                         \
    do {                                                                        \
        VPIStatus _st = (stmt);                                                 \
        if (_st != VPI_SUCCESS) {                                               \
            char msg[512];                                                      \
            vpiGetLastStatusMessage(msg, sizeof(msg));                          \
            fprintf(stderr, "[vpi_of] VPI error: %s at %s\n",                    \
                    vpiStatusGetName(_st), #stmt);                              \
            fprintf(stderr, "[vpi_of] Message: %s\n", msg);                     \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

static inline float s10_5_to_px(int16_t v) { return float(v) / 32.0f; }

static int max_levels_scale_half_min32(int W, int H)
{
    int levels = 1;
    int w = W, h = H;
    while (true) {
        int w2 = (w + 1) / 2;
        int h2 = (h + 1) / 2;
        if (w2 < 32 || h2 < 32) break;
        w = w2; h = h2;
        levels++;
    }
    return levels;
}

static int get_env_int(const char *k, int defv)
{
    const char *v = std::getenv(k);
    if (!v) return defv;
    return std::atoi(v);
}

static float get_env_float(const char *k, float defv)
{
    const char *v = std::getenv(k);
    if (!v) return defv;
    return std::atof(v);
}

static bool get_size_from_env(int &W, int &H)
{
    const char *ew = std::getenv("VPI_OF_W");
    const char *eh = std::getenv("VPI_OF_H");
    if (!ew || !eh) return false;
    W = std::atoi(ew);
    H = std::atoi(eh);
    return (W > 0 && H > 0);
}

// ----------------------------- State -----------------------------

struct State
{
    bool inited = false;
    int W = 0, H = 0;

    VPIStream stream = nullptr;

    VPIImage prev_y8_pl = nullptr;
    VPIImage cur_y8_pl  = nullptr;

    VPIPyramid prev_pyr_pl = nullptr;
    VPIPyramid cur_pyr_pl  = nullptr;
    VPIPyramid prev_pyr_bl = nullptr;
    VPIPyramid cur_pyr_bl  = nullptr;

    VPIImage *prevLvlPL = nullptr;
    VPIImage *curLvlPL  = nullptr;
    VPIImage *prevLvlBL = nullptr;
    VPIImage *curLvlBL  = nullptr;

    int numLevels = 0;
    float pyrScale = 0.5f;

    VPIPayload ofa_payload = nullptr;
    int grid = 2;

    VPIImage mv_bl     = nullptr;
    VPIImage mv_vic_pl = nullptr;
    VPIImage mv_cpu    = nullptr;

    bool havePrev = false;
    uint64_t frameCount = 0;

    uint32_t printEveryNFrames = 30;
    std::chrono::steady_clock::time_point t0, tLast;
    bool timersInit = false;

    bool overlayEnabled = true;
    int overlayStep = 12;
    float overlayScale = 3.5f;
    uint8_t overlayColorY = 235;
    int overlayMaxArrows = 4096;

    // MV debug printing
    bool mvPrintEnabled = false;
    uint32_t mvPrintEveryNFrames = 30;
    int mvPrintTopK = 8;

    // MV sanity / scene-cut handling
    float lastP50 = 0.0f;
    bool  lastP50Valid = false;

    bool warnedMissingCaps = false;
};

static State* get_or_create_state(void **userPtr)
{
    if (userPtr && *userPtr) return reinterpret_cast<State*>(*userPtr);
    State *st = new State();
    if (userPtr) *userPtr = st;
    return st;
}

static void init_vpi_if_needed(State *st, int W, int H)
{
    if (st->inited) return;

    st->W = W; st->H = H;

    st->printEveryNFrames = (uint32_t)std::max(1, get_env_int("VPI_OF_PRINT_EVERY_N", 30));
    st->overlayEnabled    = (get_env_int("VPI_OF_OVERLAY", 1) != 0);
    st->overlayStep       = std::max(1, get_env_int("VPI_OF_OVERLAY_STEP", 12));
    st->overlayScale      = std::max(0.1f, get_env_float("VPI_OF_OVERLAY_SCALE", 3.5f));
    st->overlayColorY     = (uint8_t)std::clamp(get_env_int("VPI_OF_OVERLAY_COLOR_Y", 235), 0, 255);
    st->overlayMaxArrows  = std::max(128, get_env_int("VPI_OF_OVERLAY_MAX_ARROWS", 4096));

    st->mvPrintEnabled      = (get_env_int("VPI_OF_MV_PRINT", 0) != 0);
    st->mvPrintEveryNFrames = (uint32_t)std::max(1, get_env_int("VPI_OF_MV_PRINT_EVERY_N", (int)st->printEveryNFrames));
    st->mvPrintTopK         = std::max(1, get_env_int("VPI_OF_MV_PRINT_TOPK", 8));

    CHECK_VPI(vpiStreamCreate(0, &st->stream));

    // Allocate Y8 images with CUDA backend so we can write from CUDA-EGL copy.
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &st->prev_y8_pl));
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &st->cur_y8_pl));

    st->numLevels = max_levels_scale_half_min32(W, H);

    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER,
                              st->numLevels, st->pyrScale, 0, &st->prev_pyr_pl));
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER,
                              st->numLevels, st->pyrScale, 0, &st->cur_pyr_pl));

    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER_BL,
                              st->numLevels, st->pyrScale, 0, &st->prev_pyr_bl));
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER_BL,
                              st->numLevels, st->pyrScale, 0, &st->cur_pyr_bl));

    st->prevLvlPL = new VPIImage[st->numLevels]();
    st->curLvlPL  = new VPIImage[st->numLevels]();
    st->prevLvlBL = new VPIImage[st->numLevels]();
    st->curLvlBL  = new VPIImage[st->numLevels]();

    for (int lvl = 0; lvl < st->numLevels; ++lvl) {
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(st->prev_pyr_pl, lvl, &st->prevLvlPL[lvl]));
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(st->cur_pyr_pl,  lvl, &st->curLvlPL[lvl]));
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(st->prev_pyr_bl, lvl, &st->prevLvlBL[lvl]));
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(st->cur_pyr_bl,  lvl, &st->curLvlBL[lvl]));
    }

    std::vector<int32_t> gridArr(st->numLevels, st->grid);

    CHECK_VPI(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA,
                                       W, H,
                                       VPI_IMAGE_FORMAT_Y8_ER_BL,
                                       gridArr.data(), st->numLevels,
                                       VPI_OPTICAL_FLOW_QUALITY_HIGH,
                                       &st->ofa_payload));

    const int mvW = (W + st->grid - 1) / st->grid;
    const int mvH = (H + st->grid - 1) / st->grid;

    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16_BL, 0, &st->mv_bl));
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16,
                             VPI_BACKEND_VIC | VPI_BACKEND_CPU, &st->mv_vic_pl));
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16,
                             VPI_BACKEND_CPU, &st->mv_cpu));

    st->inited = true;

    fprintf(stderr, "[vpi_of] init: %dx%d levels=%d grid=%d mv=%dx%d\n",
            W, H, st->numLevels, st->grid, mvW, mvH);

    fprintf(stderr,
            "[vpi_of] controls: printEveryN=%u overlay=%d step=%d scale=%.2f colorY=%u maxArrows=%d\n",
            st->printEveryNFrames, st->overlayEnabled ? 1 : 0, st->overlayStep,
            st->overlayScale, (unsigned)st->overlayColorY, st->overlayMaxArrows);

    fprintf(stderr, "[vpi_of] mvPrint: enabled=%d everyN=%u topK=%d\n",
            st->mvPrintEnabled ? 1 : 0, st->mvPrintEveryNFrames, st->mvPrintTopK);
}

static int build_arrow_list_cpu(State *st, std::vector<Arrow> &arrows)
{
    arrows.clear();
    arrows.reserve((size_t)st->overlayMaxArrows);

    const int mvW = (st->W + st->grid - 1) / st->grid;
    const int mvH = (st->H + st->grid - 1) / st->grid;

    VPIImageData data;
    std::memset(&data, 0, sizeof(data));

    CHECK_VPI(vpiImageLockData(st->mv_cpu, VPI_LOCK_READ,
                              VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

    const uint8_t *base = (const uint8_t*)data.buffer.pitch.planes[0].data;
    const int pitch     = (int)data.buffer.pitch.planes[0].pitchBytes;

    int x0 = (int)(mvW * 0.15f), x1 = (int)(mvW * 0.85f);
    int y0 = (int)(mvH * 0.20f), y1 = (int)(mvH * 0.80f);

    const int step = st->overlayStep;
    const float scale = st->overlayScale;
    const float maxLenPx = 80.0f;

    for (int my = y0; my < y1; my += step) {
        const int16_t *row = (const int16_t *)(base + my * pitch);
        for (int mx = x0; mx < x1; mx += step) {
            int16_t fx = row[mx * 2 + 0];
            int16_t fy = row[mx * 2 + 1];

            float dx = s10_5_to_px(fx);
            float dy = s10_5_to_px(fy);

            float mag = std::abs(dx) + std::abs(dy);
            if (mag < 0.30f) continue;

            float clx = std::clamp(dx, -maxLenPx, maxLenPx);
            float cly = std::clamp(dy, -maxLenPx, maxLenPx);

            int px0 = mx * st->grid;
            int py0 = my * st->grid;
            int px1 = px0 + (int)lrintf(clx * scale);
            int py1 = py0 + (int)lrintf(cly * scale);

            px1 = std::clamp(px1, 0, st->W - 1);
            py1 = std::clamp(py1, 0, st->H - 1);

            arrows.push_back({px0, py0, px1, py1});
            if ((int)arrows.size() >= st->overlayMaxArrows) break;
        }
        if ((int)arrows.size() >= st->overlayMaxArrows) break;
    }

    CHECK_VPI(vpiImageUnlock(st->mv_cpu));
    return (int)arrows.size();
}

// ----------------------------- MV Debug Print -----------------------------

struct MVTop
{
    float mag;
    int mx, my;
    float dx, dy;
};

static void mv_debug_print(State *st)
{
    if (!st->mvPrintEnabled) return;
    if (st->frameCount == 0) return;
    if ((st->frameCount % st->mvPrintEveryNFrames) != 0) return;

    const int mvW = (st->W + st->grid - 1) / st->grid;
    const int mvH = (st->H + st->grid - 1) / st->grid;

    VPIImageData data;
    std::memset(&data, 0, sizeof(data));

    CHECK_VPI(vpiImageLockData(st->mv_cpu, VPI_LOCK_READ,
                              VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

    const uint8_t *base = (const uint8_t*)data.buffer.pitch.planes[0].data;
    const int pitch     = (int)data.buffer.pitch.planes[0].pitchBytes;

    int x0 = (int)(mvW * 0.15f), x1 = (int)(mvW * 0.85f);
    int y0 = (int)(mvH * 0.20f), y1 = (int)(mvH * 0.80f);

    const int step = std::max(1, st->overlayStep);

    double sumMag = 0.0, sumMag2 = 0.0;
    int count = 0;
    float minMag = 1e9f, maxMag = 0.0f;

    std::vector<float> mags;
    mags.reserve(4096);

    std::vector<MVTop> top;
    top.reserve((size_t)st->mvPrintTopK);

    auto push_topk = [&](float mag, int mx, int my, float dx, float dy) {
        MVTop t{mag, mx, my, dx, dy};
        if ((int)top.size() < st->mvPrintTopK) { top.push_back(t); return; }
        int imin = 0;
        float mmin = top[0].mag;
        for (int i = 1; i < (int)top.size(); ++i) {
            if (top[i].mag < mmin) { mmin = top[i].mag; imin = i; }
        }
        if (mag > mmin) top[imin] = t;
    };

    for (int my = y0; my < y1; my += step) {
        const int16_t *row = (const int16_t *)(base + my * pitch);
        for (int mx = x0; mx < x1; mx += step) {
            int16_t fx = row[mx * 2 + 0];
            int16_t fy = row[mx * 2 + 1];

            float dx = s10_5_to_px(fx);
            float dy = s10_5_to_px(fy);

            float mag = std::sqrt(dx*dx + dy*dy);
            if (mag < 0.05f) continue;

            sumMag += mag;
            sumMag2 += (double)mag * (double)mag;
            count++;

            minMag = std::min(minMag, mag);
            maxMag = std::max(maxMag, mag);

            if ((int)mags.size() < 4096) mags.push_back(mag);

            push_topk(mag, mx, my, dx, dy);
        }
    }

    std::sort(top.begin(), top.end(), [](const MVTop &a, const MVTop &b){ return a.mag > b.mag; });

    float mean = 0.f, stdev = 0.f;
    float p50 = 0.f, p90 = 0.f, p99 = 0.f;

    if (count > 0) {
        mean = (float)(sumMag / (double)count);
        double var = (sumMag2 / (double)count) - (double)mean * (double)mean;
        if (var < 0.0) var = 0.0;
        stdev = (float)std::sqrt(var);
    }

    if (!mags.empty()) {
        auto getp = [&](float q)->float {
            size_t idx = (size_t)std::clamp((int)std::floor(q * (mags.size() - 1)), 0, (int)mags.size() - 1);
            std::nth_element(mags.begin(), mags.begin() + idx, mags.end());
            return mags[idx];
        };
        p50 = getp(0.50f);
        p90 = getp(0.90f);
        p99 = getp(0.99f);
    }

    fprintf(stderr,
            "[vpi_of][mv] frame=%llu samples=%d mag(min/mean/std/p50/p90/p99/max)=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f\n",
            (unsigned long long)st->frameCount, count,
            (count > 0 ? minMag : 0.f), mean, stdev, p50, p90, p99, (count > 0 ? maxMag : 0.f));

    for (int i = 0; i < (int)top.size(); ++i) {
        const MVTop &t = top[i];
        int px = t.mx * st->grid;
        int py = t.my * st->grid;
        fprintf(stderr,
                "[vpi_of][mv]  top[%d] mag=%.3f dx=%.3f dy=%.3f mv=(%d,%d) px=(%d,%d)\n",
                i, t.mag, t.dx, t.dy, t.mx, t.my, px, py);
    }

    CHECK_VPI(vpiImageUnlock(st->mv_cpu));
}

// ----------------------------- MV Sanity Gate -----------------------------

struct MVStats
{
    int   samples = 0;
    float p50 = 0.f;
    float p90 = 0.f;
    float p99 = 0.f;
    float maxv = 0.f;
};

static MVStats mv_compute_stats(State *st)
{
    MVStats s;

    const int mvW = (st->W + st->grid - 1) / st->grid;
    const int mvH = (st->H + st->grid - 1) / st->grid;

    VPIImageData data;
    std::memset(&data, 0, sizeof(data));

    CHECK_VPI(vpiImageLockData(st->mv_cpu, VPI_LOCK_READ,
                              VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

    const uint8_t *base = (const uint8_t*)data.buffer.pitch.planes[0].data;
    const int pitch     = (int)data.buffer.pitch.planes[0].pitchBytes;

    int x0 = (int)(mvW * 0.15f), x1 = (int)(mvW * 0.85f);
    int y0 = (int)(mvH * 0.20f), y1 = (int)(mvH * 0.80f);

    const int step = std::max(1, st->overlayStep);

    std::vector<float> mags;
    mags.reserve(4096);

    float maxMag = 0.f;

    for (int my = y0; my < y1; my += step) {
        const int16_t *row = (const int16_t *)(base + my * pitch);
        for (int mx = x0; mx < x1; mx += step) {
            int16_t fx = row[mx * 2 + 0];
            int16_t fy = row[mx * 2 + 1];

            float dx = s10_5_to_px(fx);
            float dy = s10_5_to_px(fy);

            float mag = std::sqrt(dx*dx + dy*dy);
            if (mag < 0.05f) continue;
            mags.push_back(mag);
            if (mag > maxMag) maxMag = mag;

            if ((int)mags.size() >= 4096) break;
        }
        if ((int)mags.size() >= 4096) break;
    }

    s.samples = (int)mags.size();
    s.maxv = maxMag;

    auto getp = [&](float q)->float {
        if (mags.empty()) return 0.f;
        size_t idx = (size_t)std::clamp((int)std::floor(q * (mags.size() - 1)), 0, (int)mags.size() - 1);
        std::nth_element(mags.begin(), mags.begin() + idx, mags.end());
        return mags[idx];
    };

    s.p50 = getp(0.50f);
    s.p90 = getp(0.90f);
    s.p99 = getp(0.99f);

    CHECK_VPI(vpiImageUnlock(st->mv_cpu));
    return s;
}

static bool mv_is_bad_frame(State *st, const MVStats &ms)
{
    if (ms.samples < 50) return true;

    // Absolute sanity: if typical motion is enormous -> discard.
    if (ms.p50 > 20.0f) return true;

    // Relative jump sanity: sudden jump vs previous frame.
    if (st->lastP50Valid) {
        float prev = std::max(0.05f, st->lastP50);
        if (ms.p50 > prev * 6.0f && ms.p50 > 4.0f) return true;
    }
    return false;
}

static void gpu_process(EGLImageKHR image, void **userPtr)
{
    State *st = get_or_create_state(userPtr);

    static bool cu_inited = false;
    if (!cu_inited) { cuInit(0); cu_inited = true; }

    // Size from caps OR env fallback
    int W = st->W, H = st->H;
    if (W <= 0 || H <= 0) {
        if (!get_size_from_env(W, H)) {
            if (!st->warnedMissingCaps) {
                fprintf(stderr, "[vpi_of] Missing caps W/H. Set env VPI_OF_W/VPI_OF_H.\n");
                st->warnedMissingCaps = true;
            }
            return;
        }
        st->W = W; st->H = H;
    }

    init_vpi_if_needed(st, W, H);

    // Ingest: copy NV12 Y from EGLImage -> cur_y8_pl (CUDA pitch)
    VPIImageData ydata;
    std::memset(&ydata, 0, sizeof(ydata));

    CHECK_VPI(vpiImageLockData(st->cur_y8_pl, VPI_LOCK_WRITE,
                              VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &ydata));

    uint8_t *dstY = (uint8_t*)ydata.buffer.pitch.planes[0].data;
    int dstPitch  = (int)ydata.buffer.pitch.planes[0].pitchBytes;

    egl_nv12_copy_y_to_cuda(image, dstY, dstPitch, W, H);

    CHECK_VPI(vpiImageUnlock(st->cur_y8_pl));

    if (!st->havePrev) {
        std::swap(st->prev_y8_pl, st->cur_y8_pl);
        st->havePrev = true;

        st->t0 = std::chrono::steady_clock::now();
        st->tLast = st->t0;
        st->timersInit = true;
        return;
    }

    // Phase A (CUDA): pyramids
    CHECK_VPI(vpiSubmitGaussianPyramidGenerator(st->stream, VPI_BACKEND_CUDA,
                                               st->prev_y8_pl, st->prev_pyr_pl, VPI_BORDER_CLAMP));
    CHECK_VPI(vpiSubmitGaussianPyramidGenerator(st->stream, VPI_BACKEND_CUDA,
                                               st->cur_y8_pl,  st->cur_pyr_pl,  VPI_BORDER_CLAMP));
    CHECK_VPI(vpiStreamSync(st->stream));

    // Phase B (VIC): per-level PL->BL conversions
    for (int lvl = 0; lvl < st->numLevels; ++lvl) {
        CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                             st->prevLvlPL[lvl], st->prevLvlBL[lvl], nullptr));
        CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                             st->curLvlPL[lvl],  st->curLvlBL[lvl],  nullptr));
    }
    CHECK_VPI(vpiStreamSync(st->stream));

    // Phase C (OFA): dense optical flow
    CHECK_VPI(vpiSubmitOpticalFlowDensePyramid(st->stream, VPI_BACKEND_OFA,
                                              st->ofa_payload,
                                              st->prev_pyr_bl, st->cur_pyr_bl,
                                              st->mv_bl));
    st->frameCount++;
    CHECK_VPI(vpiStreamSync(st->stream));

    const bool doPrint = (st->printEveryNFrames > 0) && ((st->frameCount % st->printEveryNFrames) == 0);
    const bool needMVcpu = st->overlayEnabled || doPrint || st->mvPrintEnabled;

    if (needMVcpu) {
        CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                             st->mv_bl, st->mv_vic_pl, nullptr));
        CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_CPU,
                                             st->mv_vic_pl, st->mv_cpu, nullptr));
        CHECK_VPI(vpiStreamSync(st->stream));

        // Debug print (optional)
        mv_debug_print(st);

        // Sanity gate: detect broken MV segments and reset history
        MVStats ms = mv_compute_stats(st);
        bool bad = mv_is_bad_frame(st, ms);

        st->lastP50 = ms.p50;
        st->lastP50Valid = true;

        if (bad) {
            fprintf(stderr, "[vpi_of][mv] BAD frame detected (p50=%.3f p99=%.3f max=%.3f) -> skip overlay + reset prev\n",
                    ms.p50, ms.p99, ms.maxv);

            std::swap(st->prev_y8_pl, st->cur_y8_pl);
            return;
        }
    }

    if (st->overlayEnabled) {
        std::vector<Arrow> arrows;
        int n = build_arrow_list_cpu(st, arrows);
        if (n > 0) {
            overlay_draw_arrows_nv12(image, W, H, arrows.data(), n, st->overlayColorY);
        }
    }

    if (!st->timersInit) {
        st->t0 = std::chrono::steady_clock::now();
        st->tLast = st->t0;
        st->timersInit = true;
    }

    if (doPrint) {
        auto now = std::chrono::steady_clock::now();
        double secFromStart = std::chrono::duration<double>(now - st->t0).count();
        double secFromLast  = std::chrono::duration<double>(now - st->tLast).count();
        double fpsWindow    = double(st->printEveryNFrames) / std::max(1e-9, secFromLast);

        fprintf(stderr, "[vpi_of] frames=%llu elapsed=%.2fs fps~%.1f\n",
                (unsigned long long)st->frameCount, secFromStart, fpsWindow);
        st->tLast = now;
    }

    std::swap(st->prev_y8_pl, st->cur_y8_pl);
}

static void pre_process(void **, unsigned int *inW, unsigned int *inH,
                        unsigned int*, unsigned int*, ColorFormat*,
                        unsigned int, void **userPtr)
{
    State *st = get_or_create_state(userPtr);
    if (inW && inH && *inW > 0 && *inH > 0) {
        st->W = (int)*inW;
        st->H = (int)*inH;
    }
}

static void post_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*,
                         ColorFormat*, unsigned int, void **)
{
    // don't free here
}

extern "C" void init(CustomerFunction *f)
{
    if (!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;

    fprintf(stderr, "[vpi_of] init(): VPI OFA + CUDA overlay loaded\n");
    fprintf(stderr, "[vpi_of] NOTE: If caps W/H are missing, set env VPI_OF_W/VPI_OF_H\n");
}

extern "C" void deinit(void)
{
    fprintf(stderr, "[vpi_of] deinit(): called\n");
}
