// nvivafilter_vpi_of.cpp
//
// Jetson Orin / JetPack 5.4.1 / VPI 2.4.x
//
// NVDEC -> NVMM(NV12) -> nvivafilter (this .so) -> NVMM
//
// VPI 2.4 + OFA Dense Optical Flow.
// OVERLAY:
// - Draws motion-vector arrows (red) onto NV12 via CUDA EGL interop.
// - Draws a single "resultant" arrow (blue) at the image center.
// OUTPUT:
// - Prints ONLY speed per frame to stdout, based on:
//     1 meter = 717 pixels, FPS = 30
//   Speed formula: m/s = (px_per_frame * FPS) / 717
//
// Stability fix:
// - DO NOT wrap EGLImage into VPI on nvbuf-mem-surface-array pipelines.
// - Some pipelines keep the NVMM container locked exclusively during the callback.
// - Use CUDA-EGL to copy NV12 Y plane into a CUDA-backed VPI image, then operate only on VPI-owned buffers.

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

// ----------------------------- MV stats helpers -----------------------------

struct MVStats
{
    int samples = 0;
    float p50 = 0.0f;
    float p90 = 0.0f;
    float p99 = 0.0f;
    float maxv = 0.0f;
    float mean = 0.0f;
    float stdv = 0.0f;

    // Vector-mean (resultant motion) in px/frame.
    float mean_dx = 0.0f;
    float mean_dy = 0.0f;
    float res_mag = 0.0f;  // hypot(mean_dx, mean_dy)
};

// Compute percentile via nth_element (no full sort)
static float percentile_inplace(std::vector<float> &v, float q01)
{
    if (v.empty()) return 0.0f;
    q01 = std::clamp(q01, 0.0f, 1.0f);
    size_t k = (size_t)std::llround((v.size() - 1) * q01);
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

struct MVTop
{
    float mag;
    float dx, dy;
    int mx, my;
};

static MVStats compute_mv_stats_and_topk(
    VPIImage mv_cpu,
    int mvW, int mvH,
    int x0, int x1, int y0, int y1,
    int step,
    int topK,
    std::vector<MVTop> &top)
{
    MVStats st;
    top.clear();
    top.reserve((size_t)topK);

    VPIImageData data;
    std::memset(&data, 0, sizeof(data));
    CHECK_VPI(vpiImageLockData(mv_cpu, VPI_LOCK_READ,
                              VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

    const uint8_t *base = (const uint8_t*)data.buffer.pitch.planes[0].data;
    const int pitch     = (int)data.buffer.pitch.planes[0].pitchBytes;

    std::vector<float> mags;
    mags.reserve((size_t)((((x1-x0)+step-1)/step) * (((y1-y0)+step-1)/step)));

    double sum  = 0.0;
    double sum2 = 0.0;
    double sum_dx = 0.0;
    double sum_dy = 0.0;

    float maxv = 0.0f;

    auto top_push = [&](float mag, float dx, float dy, int mx, int my) {
        if ((int)top.size() < topK) {
            top.push_back({mag, dx, dy, mx, my});
            if ((int)top.size() == topK) {
                std::sort(top.begin(), top.end(),
                          [](const MVTop &a, const MVTop &b){ return a.mag > b.mag; });
            }
            return;
        }
        if (mag <= top.back().mag) return;
        top.back() = {mag, dx, dy, mx, my};
        // Bubble down (keep sorted descending)
        for (int i = topK - 1; i > 0; --i) {
            if (top[i].mag > top[i-1].mag) std::swap(top[i], top[i-1]);
            else break;
        }
    };

    for (int my = y0; my < y1; my += step) {
        const int16_t *row = (const int16_t *)(base + my * pitch);
        for (int mx = x0; mx < x1; mx += step) {
            int16_t fx = row[mx * 2 + 0];
            int16_t fy = row[mx * 2 + 1];
            float dx = s10_5_to_px(fx);
            float dy = s10_5_to_px(fy);
            float mag = std::hypot(dx, dy);

            mags.push_back(mag);
            st.samples++;

            sum  += mag;
            sum2 += (double)mag * (double)mag;
            sum_dx += dx;
            sum_dy += dy;

            maxv = std::max(maxv, mag);

            top_push(mag, dx, dy, mx, my);
        }
    }

    CHECK_VPI(vpiImageUnlock(mv_cpu));

    st.maxv = maxv;
    if (st.samples > 0) {
        st.mean = (float)(sum / (double)st.samples);
        double var = (sum2 / (double)st.samples) -
                     ((sum / (double)st.samples) * (sum / (double)st.samples));
        st.stdv = (float)std::sqrt(std::max(0.0, var));

        st.mean_dx = (float)(sum_dx / (double)st.samples);
        st.mean_dy = (float)(sum_dy / (double)st.samples);
        st.res_mag = (float)std::hypot(st.mean_dx, st.mean_dy);

        st.p50 = percentile_inplace(mags, 0.50f);
        st.p90 = percentile_inplace(mags, 0.90f);
        st.p99 = percentile_inplace(mags, 0.99f);
    }
    return st;
}

static bool is_bad_frame_spike(const MVStats &s, float ratio, float minAbs)
{
    // Spike-based rule:
    // - Allow any global motion magnitude, reject only frames where
    //   the tail explodes relative to the median/upper quantiles.
    // - minAbs prevents tiny-noise cases from triggering on ratios.
    if (s.samples <= 0) return false;

    float p50 = std::max(1e-6f, s.p50);
    float p90 = std::max(1e-6f, s.p90);

    bool spike_tail = (s.p99 > p50 * ratio) || (s.maxv > p90 * ratio);
    bool strong_enough = (s.maxv >= minAbs);

    return spike_tail && strong_enough;
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

    // Overlay controls
    bool overlayEnabled = true;
    int overlayStep = 12;
    float overlayScale = 3.5f;
    int overlayMaxArrows = 4096;

    // MV filtering (optional safety)
    float badRatio = 3.0f;    // tail ratio threshold
    float badMinAbs = 60.0f;  // minimum absolute max to consider a spike "real"

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

    // Runtime controls via environment variables.
    st->overlayEnabled    = (get_env_int("VPI_OF_OVERLAY", 1) != 0);
    st->overlayStep       = std::max(1, get_env_int("VPI_OF_OVERLAY_STEP", 12));
    st->overlayScale      = std::max(0.1f, get_env_float("VPI_OF_OVERLAY_SCALE", 3.5f));
    st->overlayMaxArrows  = std::max(128, get_env_int("VPI_OF_OVERLAY_MAX_ARROWS", 4096));

    st->badRatio  = std::max(1.2f, get_env_float("VPI_OF_BAD_RATIO", 3.0f));
    st->badMinAbs = std::max(0.0f, get_env_float("VPI_OF_BAD_MIN_ABS", 60.0f));

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
}

struct Arrow;
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

    // Center ROI used for arrows too (matches stats)
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

static inline void print_speed_only(const MVStats &s)
{
    // Convert resultant motion from px/frame to m/s and km/h.
    // Given: 1 meter = 717 pixels, FPS = 30.
    constexpr float PX_PER_M = 717.0f;
    constexpr float FPS = 30.0f;

    float speed_mps = (s.res_mag * FPS) / PX_PER_M;
    float speed_kmh = speed_mps * 3.6f;

    // Print ONLY speed per frame (no other logs).
    std::fprintf(stdout, "%.3f m/s (%.2f km/h)\n", speed_mps, speed_kmh);
    std::fflush(stdout);
}

static inline void print_speed_zero()
{
    std::fprintf(stdout, "0.000 m/s (0.00 km/h)\n");
    std::fflush(stdout);
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
            // No other prints allowed; just skip silently.
            return;
        }
        st->W = W; st->H = H;
    }

    init_vpi_if_needed(st, W, H);

    // --------------------------------------------------------------------
    // Ingest: copy NV12 luma (Y) from EGLImage -> cur_y8_pl (CUDA memory)
    // --------------------------------------------------------------------
    VPIImageData ydata;
    std::memset(&ydata, 0, sizeof(ydata));

    CHECK_VPI(vpiImageLockData(st->cur_y8_pl, VPI_LOCK_WRITE,
                              VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &ydata));

    uint8_t *dstY = (uint8_t*)ydata.buffer.pitch.planes[0].data;
    int dstPitch  = (int)ydata.buffer.pitch.planes[0].pitchBytes;

    egl_nv12_copy_y_to_cuda(image, dstY, dstPitch, W, H);

    CHECK_VPI(vpiImageUnlock(st->cur_y8_pl));

    // First frame: no previous frame available -> speed is 0
    if (!st->havePrev) {
        print_speed_zero();
        std::swap(st->prev_y8_pl, st->cur_y8_pl);
        st->havePrev = true;
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

    // Always compute stats/speed every frame -> need CPU MV.
    CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                         st->mv_bl, st->mv_vic_pl, nullptr));
    CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_CPU,
                                         st->mv_vic_pl, st->mv_cpu, nullptr));
    CHECK_VPI(vpiStreamSync(st->stream));

    const int mvW = (W + st->grid - 1) / st->grid;
    const int mvH = (H + st->grid - 1) / st->grid;

    int x0 = (int)(mvW * 0.15f), x1 = (int)(mvW * 0.85f);
    int y0 = (int)(mvH * 0.20f), y1 = (int)(mvH * 0.80f);

    std::vector<MVTop> topDummy;
    MVStats s = compute_mv_stats_and_topk(
        st->mv_cpu, mvW, mvH,
        x0, x1, y0, y1,
        st->overlayStep, /*topK*/1, topDummy);

    bool badFrame = is_bad_frame_spike(s, st->badRatio, st->badMinAbs);
    if (badFrame) {
        // Print 0 for this frame, reseed previous, skip overlay.
        print_speed_zero();
        std::swap(st->prev_y8_pl, st->cur_y8_pl);
        st->havePrev = true;
        return;
    }

    // Print speed for this frame (ONLY output allowed).
    print_speed_only(s);

    // Overlay:
    // - Red vector field
    // - Blue resultant arrow at image center
    if (st->overlayEnabled) {
        // Red field arrows
        std::vector<Arrow> arrows;
        int n = build_arrow_list_cpu(st, arrows);
        if (n > 0) {
            overlay_draw_arrows_nv12(image, W, H, arrows.data(), n,
                                     /*Y*/76, /*U*/85, /*V*/255);
        }

        // Blue resultant at center
        int cx = W / 2;
        int cy = H / 2;

        float dx = s.mean_dx;
        float dy = s.mean_dy;

        float sc = st->overlayScale;
        int x1p = cx + (int)lrintf(dx * sc);
        int y1p = cy + (int)lrintf(dy * sc);

        x1p = std::clamp(x1p, 0, W - 1);
        y1p = std::clamp(y1p, 0, H - 1);

        Arrow res = {cx, cy, x1p, y1p};
        overlay_draw_arrows_nv12(image, W, H, &res, 1,
                                 /*Blue (approx)*/29, 255, 107);
    }

    // Normal advance
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
    // Intentionally do not free here.
}

extern "C" void init(CustomerFunction *f)
{
    if (!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;

    // No prints here (requirement: only speed per frame).
}

extern "C" void deinit(void)
{
    // No prints here (requirement: only speed per frame).
}
