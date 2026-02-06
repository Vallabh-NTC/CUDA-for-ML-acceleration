// nvivafilter_vpi_of.cpp
//
// Jetson Orin / JetPack 5.4.1 / VPI 2.4.x
//
// NVDEC -> NVMM(NV12) -> nvivafilter (this .so) -> NVMM
//
// VPI2.4 EGL wrapper (NO CUDA-EGL mapping) + OFA Dense Optical Flow.
// Prints motion vectors (dx,dy) in pixels every ~1 second (every 30 frames).
//
// IMPORTANT (why we do an extra CPU copy):
// - OFA output is block-linear (2S16_BL). CPU can't read it directly.
// - We convert BL -> pitch-linear using VIC into a VIC-enabled container (mv_vic_pl).
// - Then we copy to a CPU-enabled container (mv_cpu) using CPU backend,
//   ONLY for printing/debug (can be removed later for full GPU-only overlay).
//
// Width/Height:
// - Prefer caps via pre_process().
// - Fallback to env VPI_OF_W / VPI_OF_H.
//
// Assumptions about MV format:
// - 2S16 values are treated as S10.5 fixed-point => pixels = value / 32.0

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>

#include <EGL/egl.h>
#include <cuda.h> // only for cuInit safety

#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Pyramid.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/OpticalFlowDense.h>

#include "nvivafilter_customer_api.hpp"

// ----------------------------- Helpers -----------------------------

#define CHECK_VPI(stmt)                                                         \
    do {                                                                        \
        VPIStatus _st = (stmt);                                                 \
        if (_st != VPI_SUCCESS) {                                               \
            char msg[512];                                                      \
            vpiGetLastStatusMessage(msg, sizeof(msg));                          \
            fprintf(stderr, "[vpi_of] VPI error: %s at %s\n",                   \
                    vpiStatusGetName(_st), #stmt);                              \
            fprintf(stderr, "[vpi_of] Message: %s\n", msg);                     \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

static inline float s10_5_to_px(int16_t v) { return float(v) / 32.0f; }

// Compute max pyramid levels for scale=0.5 such that smallest level >= 32x32
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

static bool get_size_from_env(int &W, int &H)
{
    const char* ew = std::getenv("VPI_OF_W");
    const char* eh = std::getenv("VPI_OF_H");
    if (!ew || !eh) return false;
    W = std::atoi(ew);
    H = std::atoi(eh);
    return (W > 0 && H > 0);
}

// ----------------------------- State -----------------------------

struct State
{
    bool inited = false;

    int W = 0;
    int H = 0;

    VPIStream stream = nullptr;

    // Wrapped input (NV12 as EGLImage wrapper)
    VPIImage vpi_in_nv12 = nullptr;

    // Y8 pitch-linear frames
    VPIImage prev_y8_pl = nullptr;
    VPIImage cur_y8_pl  = nullptr;

    // Pyramids
    VPIPyramid prev_pyr_pl = nullptr;
    VPIPyramid cur_pyr_pl  = nullptr;
    VPIPyramid prev_pyr_bl = nullptr;
    VPIPyramid cur_pyr_bl  = nullptr;

    // Level wrappers
    VPIImage *prevLvlPL = nullptr;
    VPIImage *curLvlPL  = nullptr;
    VPIImage *prevLvlBL = nullptr;
    VPIImage *curLvlBL  = nullptr;

    int numLevels = 0;
    float pyrScale = 0.5f;

    // OFA
    VPIPayload ofa_payload = nullptr;
    int grid = 2;

    // OFA output and debug copies
    VPIImage mv_bl     = nullptr; // 2S16_BL (OFA output)
    VPIImage mv_vic_pl = nullptr; // 2S16 pitch-linear, VIC-enabled (conversion target)
    VPIImage mv_cpu    = nullptr; // 2S16 pitch-linear, CPU-enabled (for vpiImageLockData)

    bool havePrev = false;

    // Debug counters
    uint64_t frameCount = 0;
    std::chrono::steady_clock::time_point t0;
    std::chrono::steady_clock::time_point tLast;
    bool timersInit = false;
};

static State* get_or_create_state(void **userPtr)
{
    if (userPtr && *userPtr) return reinterpret_cast<State*>(*userPtr);
    State *st = new State();
    if (userPtr) *userPtr = st;
    return st;
}

static void destroy_state(State *st)
{
    if (!st) return;

    if (st->prevLvlPL) {
        for (int i=0;i<st->numLevels;i++) if (st->prevLvlPL[i]) vpiImageDestroy(st->prevLvlPL[i]);
        delete [] st->prevLvlPL;
    }
    if (st->curLvlPL) {
        for (int i=0;i<st->numLevels;i++) if (st->curLvlPL[i]) vpiImageDestroy(st->curLvlPL[i]);
        delete [] st->curLvlPL;
    }
    if (st->prevLvlBL) {
        for (int i=0;i<st->numLevels;i++) if (st->prevLvlBL[i]) vpiImageDestroy(st->prevLvlBL[i]);
        delete [] st->prevLvlBL;
    }
    if (st->curLvlBL) {
        for (int i=0;i<st->numLevels;i++) if (st->curLvlBL[i]) vpiImageDestroy(st->curLvlBL[i]);
        delete [] st->curLvlBL;
    }

    if (st->mv_cpu)    vpiImageDestroy(st->mv_cpu);
    if (st->mv_vic_pl) vpiImageDestroy(st->mv_vic_pl);
    if (st->mv_bl)     vpiImageDestroy(st->mv_bl);

    if (st->ofa_payload) vpiPayloadDestroy(st->ofa_payload);

    if (st->prev_pyr_bl) vpiPyramidDestroy(st->prev_pyr_bl);
    if (st->cur_pyr_bl)  vpiPyramidDestroy(st->cur_pyr_bl);
    if (st->prev_pyr_pl) vpiPyramidDestroy(st->prev_pyr_pl);
    if (st->cur_pyr_pl)  vpiPyramidDestroy(st->cur_pyr_pl);

    if (st->prev_y8_pl) vpiImageDestroy(st->prev_y8_pl);
    if (st->cur_y8_pl)  vpiImageDestroy(st->cur_y8_pl);

    if (st->vpi_in_nv12) vpiImageDestroy(st->vpi_in_nv12);

    if (st->stream) vpiStreamDestroy(st->stream);

    delete st;
}

// ----------------------------- VPI init -----------------------------

static void init_vpi_if_needed(State *st, int W, int H)
{
    if (st->inited) return;

    st->W = W;
    st->H = H;

    CHECK_VPI(vpiStreamCreate(0, &st->stream));

    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, 0, &st->prev_y8_pl));
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, 0, &st->cur_y8_pl));

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

    // OFA output is block-linear
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16_BL, 0, &st->mv_bl));

    // Conversion target for VIC must have VIC backend enabled
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16,
                         VPI_BACKEND_VIC | VPI_BACKEND_CPU, &st->mv_vic_pl));

    // CPU-readable container
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16, VPI_BACKEND_CPU, &st->mv_cpu));

    st->inited = true;

    fprintf(stderr, "[vpi_of] VPI initialized: %dx%d levels=%d grid=%d mv=%dx%d\n",
            W, H, st->numLevels, st->grid, mvW, mvH);
}

// ----------------------------- EGLImage -> VPI NV12 wrapper (VPI 2.4) -----------------------------

static bool vpi_wrap_input_from_eglimage(State *st, EGLImageKHR image)
{
    if (st->vpi_in_nv12) {
        vpiImageDestroy(st->vpi_in_nv12);
        st->vpi_in_nv12 = nullptr;
    }

    VPIImageData data;
    std::memset(&data, 0, sizeof(data));
    data.bufferType = VPI_IMAGE_BUFFER_EGLIMAGE;
    data.buffer.egl = image;

    VPIImageWrapperParams wparams;
    CHECK_VPI(vpiInitImageWrapperParams(&wparams));
    wparams.colorSpec = VPI_COLOR_SPEC_DEFAULT;

    CHECK_VPI(vpiImageCreateWrapper(&data, &wparams, 0, &st->vpi_in_nv12));
    return true;
}

// ----------------------------- Debug: print a few motion vectors -----------------------------

static void print_flow_samples(State *st)
{
    const int mvW = (st->W + st->grid - 1) / st->grid;
    const int mvH = (st->H + st->grid - 1) / st->grid;

    VPIImageData data;
    std::memset(&data, 0, sizeof(data));

    // VPI 2.4: lock CPU image to get HOST_PITCH_LINEAR pointer
    CHECK_VPI(vpiImageLockData(st->mv_cpu, VPI_LOCK_READ,
                              VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

    const uint8_t *base = (const uint8_t*)data.buffer.pitch.planes[0].data;
    const int pitch     = (int)data.buffer.pitch.planes[0].pitchBytes;

    auto read_vec = [&](int mx, int my, int16_t &dx, int16_t &dy) {
        const int16_t *row = (const int16_t *)(base + my * pitch);
        dx = row[mx * 2 + 0];
        dy = row[mx * 2 + 1];
    };

    int16_t dx, dy;

    // Center sample
    int cx = mvW / 2;
    int cy = mvH / 2;
    read_vec(cx, cy, dx, dy);
    fprintf(stderr,
            "[vpi_of] frame=%llu  MV(center) @ input≈(%d,%d): dx=%.2f dy=%.2f px\n",
            (unsigned long long)st->frameCount,
            cx * st->grid, cy * st->grid,
            s10_5_to_px(dx), s10_5_to_px(dy));

    // A second sample near top-left (clamped)
    int sx1 = std::min(10, mvW - 1);
    int sy1 = std::min(10, mvH - 1);
    read_vec(sx1, sy1, dx, dy);
    fprintf(stderr,
            "[vpi_of] frame=%llu  MV(10,10) @ input≈(%d,%d): dx=%.2f dy=%.2f px\n",
            (unsigned long long)st->frameCount,
            sx1 * st->grid, sy1 * st->grid,
            s10_5_to_px(dx), s10_5_to_px(dy));

    // A third sample near bottom-right (clamped)
    int sx2 = std::max(0, mvW - 11);
    int sy2 = std::max(0, mvH - 11);
    read_vec(sx2, sy2, dx, dy);
    fprintf(stderr,
            "[vpi_of] frame=%llu  MV(end-10,end-10) @ input≈(%d,%d): dx=%.2f dy=%.2f px\n",
            (unsigned long long)st->frameCount,
            sx2 * st->grid, sy2 * st->grid,
            s10_5_to_px(dx), s10_5_to_px(dy));

    CHECK_VPI(vpiImageUnlock(st->mv_cpu));
}

// ----------------------------- Main processing -----------------------------

static void gpu_process(EGLImageKHR image, void **userPtr)
{
    State *st = get_or_create_state(userPtr);

    static bool cuda_inited = false;
    if (!cuda_inited) { cuInit(0); cuda_inited = true; }

    // Prefer size from caps (pre_process), fallback to env
    int W = st->W, H = st->H;
    if (W <= 0 || H <= 0) {
        if (!get_size_from_env(W, H)) {
            fprintf(stderr, "[vpi_of] Missing size. Set env VPI_OF_W/VPI_OF_H or ensure caps reach pre_process().\n");
            return;
        }
        st->W = W;
        st->H = H;
    }

    init_vpi_if_needed(st, W, H);

    if (!vpi_wrap_input_from_eglimage(st, image)) return;

    // NV12 -> Y8 (VIC)
    CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                         st->vpi_in_nv12, st->cur_y8_pl, nullptr));

    if (!st->havePrev) {
        CHECK_VPI(vpiStreamSync(st->stream));
        std::swap(st->prev_y8_pl, st->cur_y8_pl);
        st->havePrev = true;

        st->t0 = std::chrono::steady_clock::now();
        st->tLast = st->t0;
        st->timersInit = true;
        return;
    }

    // Build pyramids (CUDA)
    CHECK_VPI(vpiSubmitGaussianPyramidGenerator(st->stream, VPI_BACKEND_CUDA,
                                               st->prev_y8_pl, st->prev_pyr_pl, VPI_BORDER_CLAMP));
    CHECK_VPI(vpiSubmitGaussianPyramidGenerator(st->stream, VPI_BACKEND_CUDA,
                                               st->cur_y8_pl,  st->cur_pyr_pl,  VPI_BORDER_CLAMP));
    CHECK_VPI(vpiStreamSync(st->stream));

    // PL -> BL (VIC)
    for (int lvl = 0; lvl < st->numLevels; ++lvl) {
        CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                             st->prevLvlPL[lvl], st->prevLvlBL[lvl], nullptr));
        CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                             st->curLvlPL[lvl],  st->curLvlBL[lvl],  nullptr));
    }
    CHECK_VPI(vpiStreamSync(st->stream));

    // OFA dense flow (output in mv_bl)
    CHECK_VPI(vpiSubmitOpticalFlowDensePyramid(st->stream, VPI_BACKEND_OFA,
                                              st->ofa_payload, st->prev_pyr_bl, st->cur_pyr_bl, st->mv_bl));
    CHECK_VPI(vpiStreamSync(st->stream));

    // Debug path (GPU -> CPU) for printing only:
    // 1) mv_bl (BL) -> mv_vic_pl (PL) using VIC
    CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                         st->mv_bl, st->mv_vic_pl, nullptr));
    CHECK_VPI(vpiStreamSync(st->stream));

    // 2) mv_vic_pl -> mv_cpu using CPU backend (so we can lock on host)
    CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_CPU,
                                         st->mv_vic_pl, st->mv_cpu, nullptr));
    CHECK_VPI(vpiStreamSync(st->stream));

    st->frameCount++;

    if (!st->timersInit) {
        st->t0 = std::chrono::steady_clock::now();
        st->tLast = st->t0;
        st->timersInit = true;
    }

    // Print every 30 frames (~1 second at 30 fps)
    if ((st->frameCount % 30) == 0) {
        auto now = std::chrono::steady_clock::now();
        double secFromStart = std::chrono::duration<double>(now - st->t0).count();
        double secFromLast  = std::chrono::duration<double>(now - st->tLast).count();
        double fpsWindow    = 30.0 / std::max(1e-9, secFromLast);

        fprintf(stderr, "[vpi_of] frames=%llu  elapsed=%.2fs  fps~%.1f\n",
                (unsigned long long)st->frameCount, secFromStart, fpsWindow);

        print_flow_samples(st);
        st->tLast = now;
    }

    // Ping-pong
    std::swap(st->prev_y8_pl, st->cur_y8_pl);
}

// ----------------------------- nvivafilter hooks -----------------------------

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
    // Not used
}

extern "C" void init(CustomerFunction *f)
{
    if (!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;

    fprintf(stderr, "[vpi_of] init(): VPI OFA filter loaded (pass-through + compute)\n");
    fprintf(stderr, "[vpi_of] NOTE: size from caps (preferred) or env VPI_OF_W/VPI_OF_H (fallback)\n");
    fprintf(stderr, "[vpi_of] NOTE: CPU copy is only for printing/debug and can be removed later.\n");
}

extern "C" void deinit(void)
{
    fprintf(stderr, "[vpi_of] deinit(): called\n");
}
