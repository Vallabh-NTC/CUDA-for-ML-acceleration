// nvivafilter_vpi_of.cpp
//
// Jetson Orin / JetPack 5.4.1 / VPI 2.4.x
//
// NVDEC -> NVMM(NV12) -> nvivafilter (this .so) -> NVMM
//
// VPI 2.4 + OFA Dense Optical Flow.
//
// Key constraint on VPI 2.4:
// - CUDA conversion is NOT implemented for 2S16_BL -> 2S16.
// - Locking OFA output (2S16_BL) as CUDA_ARRAY might not be supported.
// Therefore we use VIC to convert mv_bl -> mv_pl (2S16 pitch-linear),
// and then lock mv_pl as CUDA_PITCH_LINEAR for GPU reduction/overlay.
//
// CPU does only dt timing + stdout printing.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath>


#include <EGL/egl.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Pyramid.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/OpticalFlowDense.h>

#include "nvivafilter_customer_api.hpp"
#include "egl_copy.hpp"
#include "overlay.hpp"
#include "mv_reduce.hpp"

#define CHECK_VPI(stmt)                                                         \
    do {                                                                        \
        VPIStatus _st = (stmt);                                                 \
        if (_st != VPI_SUCCESS) {                                               \
            char msg[512];                                                      \
            vpiGetLastStatusMessage(msg, sizeof(msg));                          \
            std::fprintf(stderr, "[vpi_of] VPI error: %s at %s\n",              \
                         vpiStatusGetName(_st), #stmt);                         \
            std::fprintf(stderr, "[vpi_of] Message: %s\n", msg);                \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

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

static inline void print_speed_mps(float speed_mps)
{
    float speed_kmh = speed_mps * 3.6f;
    std::fprintf(stdout, "%.3f m/s (%.2f km/h)\n", speed_mps, speed_kmh);
    std::fflush(stdout);
}

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

    // OFA output MV in block-linear
    VPIImage mv_bl = nullptr;

    // VIC->pitch-linear MV that is also CUDA-accessible (for GPU read)
    VPIImage mv_vic_cuda_pl = nullptr;

    bool havePrev = false;

    // ROI controls in normalized coordinates [0..1] in MV space.
    // They are applied to MV grid dimensions (mvW, mvH).
    float roiX0 = 0.15f;
    float roiX1 = 0.85f;
    float roiY0 = 0.20f;
    float roiY1 = 0.80f;


    // Overlay controls
    bool overlayEnabled = true;
    int overlayStep = 12;
    float overlayScale = 3.5f;

    // Band-pass for MV magnitudes (px/frame)
    float mvMinMag = 0.5f;
    float mvMaxMag = 120.0f;

    // Gating thresholds
    int   minSamples = 64;
    float cohMin     = 0.30f;
    float stdMax     = 12.0f;
    float tailRatio  = 3.0f;
    float tailMinAbs = 60.0f;

    // Calibration
    float pxPerM = 717.0f;

    // Adaptive EMA
    float emaAlphaHi = 0.50f;
    float emaAlphaLo = 0.15f;

    // Timing
    bool haveTime = false;
    std::chrono::steady_clock::time_point lastTime;

    // GPU EMA state and speed output
    DevEmaState *d_ema = nullptr;
    float       *d_speed_out = nullptr;

    // CPU-side
    float speed_host  = 0.0f;
    float res_dx_host = 0.0f;
    float res_dy_host = 0.0f;
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

    // Env runtime controls
    st->overlayEnabled = (get_env_int("VPI_OF_OVERLAY", 1) != 0);
    st->overlayStep    = std::max(1, get_env_int("VPI_OF_OVERLAY_STEP", 12));
    st->overlayScale   = std::max(0.1f, get_env_float("VPI_OF_OVERLAY_SCALE", 3.5f));

    st->mvMinMag = std::max(0.0f, get_env_float("VPI_OF_MIN_MAG", 0.5f));
    st->mvMaxMag = std::max(st->mvMinMag + 0.1f, get_env_float("VPI_OF_MAX_MAG", 120.0f));

    st->minSamples = std::max(16, get_env_int("VPI_OF_MIN_ROBUST_SAMPLES", 64));
    st->cohMin     = std::clamp(get_env_float("VPI_OF_COH_MIN", 0.30f), 0.0f, 1.0f);
    st->stdMax     = std::max(0.0f, get_env_float("VPI_OF_STD_MAX", 12.0f));

    st->tailRatio  = std::max(1.0f, get_env_float("VPI_OF_TAIL_RATIO", 3.0f));
    st->tailMinAbs = std::max(0.0f, get_env_float("VPI_OF_TAIL_MIN_ABS", 60.0f));

    st->pxPerM = std::max(1.0f, get_env_float("VPI_OF_PX_PER_M", 717.0f));

    st->emaAlphaHi = std::clamp(get_env_float("VPI_OF_EMA_ALPHA_HI", 0.50f), 0.0f, 1.0f);
    st->emaAlphaLo = std::clamp(get_env_float("VPI_OF_EMA_ALPHA_LO", 0.15f), 0.0f, 1.0f);

    // ROI (normalized) runtime controls.
    // Example: export VPI_OF_ROI_X0=0.30 VPI_OF_ROI_X1=0.70 VPI_OF_ROI_Y0=0.35 VPI_OF_ROI_Y1=0.65
    st->roiX0 = std::clamp(get_env_float("VPI_OF_ROI_X0", 0.15f), 0.0f, 1.0f);
    st->roiX1 = std::clamp(get_env_float("VPI_OF_ROI_X1", 0.85f), 0.0f, 1.0f);
    st->roiY0 = std::clamp(get_env_float("VPI_OF_ROI_Y0", 0.20f), 0.0f, 1.0f);
    st->roiY1 = std::clamp(get_env_float("VPI_OF_ROI_Y1", 0.80f), 0.0f, 1.0f);

    // Sanity: enforce non-empty ROI, otherwise fallback to defaults.
    if (!(st->roiX1 > st->roiX0 + 0.01f)) { st->roiX0 = 0.15f; st->roiX1 = 0.85f; }
    if (!(st->roiY1 > st->roiY0 + 0.01f)) { st->roiY0 = 0.20f; st->roiY1 = 0.80f; }


    CHECK_VPI(vpiStreamCreate(0, &st->stream));

    // CUDA-backed Y8 for CUDA-EGL copy
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &st->prev_y8_pl));
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &st->cur_y8_pl));

    st->numLevels = max_levels_scale_half_min32(W, H);

    // Pyramids
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

    // OFA payload
    std::vector<int32_t> gridArr(st->numLevels, st->grid);
    CHECK_VPI(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA,
                                       W, H,
                                       VPI_IMAGE_FORMAT_Y8_ER_BL,
                                       gridArr.data(), st->numLevels,
                                       VPI_OPTICAL_FLOW_QUALITY_HIGH,
                                       &st->ofa_payload));

    // MV images
    const int mvW = (W + st->grid - 1) / st->grid;
    const int mvH = (H + st->grid - 1) / st->grid;

    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16_BL, 0, &st->mv_bl));

    // IMPORTANT: destination supports VIC (writer) + CUDA (reader)
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16,
                             VPI_BACKEND_VIC | VPI_BACKEND_CUDA, &st->mv_vic_cuda_pl));

    // Persistent GPU EMA + speed output
    cudaMalloc(&st->d_ema, sizeof(DevEmaState));
    cudaMalloc(&st->d_speed_out, sizeof(float));
    cudaMemset(st->d_ema, 0, sizeof(DevEmaState));
    cudaMemset(st->d_speed_out, 0, sizeof(float));

    st->inited = true;
}

static void gpu_process(EGLImageKHR image, void **userPtr)
{
    State *st = get_or_create_state(userPtr);

    static bool cu_inited = false;
    if (!cu_inited) { cuInit(0); cu_inited = true; }

    // Size from caps or env fallback
    int W = st->W, H = st->H;
    if (W <= 0 || H <= 0) {
        if (!get_size_from_env(W, H)) return;
        st->W = W; st->H = H;
    }

    init_vpi_if_needed(st, W, H);

    // dt on CPU
    auto now = std::chrono::steady_clock::now();
    float dtSec = 0.0f;
    if (!st->haveTime) {
        st->haveTime = true;
        st->lastTime = now;
    } else {
        dtSec = std::chrono::duration<float>(now - st->lastTime).count();
        st->lastTime = now;
        dtSec = std::clamp(dtSec, 1.0f/80.0f, 1.0f/10.0f);
    }
    float useDt = (dtSec > 0.0f) ? dtSec : (1.0f / 30.0f);

    // Copy NV12 luma (Y) from EGLImage to CUDA-backed VPI Y8
    VPIImageData ydata{};
    CHECK_VPI(vpiImageLockData(st->cur_y8_pl, VPI_LOCK_WRITE,
                              VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &ydata));

    uint8_t *dstY = (uint8_t*)ydata.buffer.pitch.planes[0].data;
    int dstPitch  = (int)ydata.buffer.pitch.planes[0].pitchBytes;

    egl_nv12_copy_y_to_cuda(image, dstY, dstPitch, W, H);

    CHECK_VPI(vpiImageUnlock(st->cur_y8_pl));

    // First frame
    if (!st->havePrev) {
        print_speed_mps(0.0f);
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

    // Phase B (VIC): PL->BL per level
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
    CHECK_VPI(vpiStreamSync(st->stream));

    // Convert MV: BL -> pitch-linear using VIC (CUDA conversion not available in VPI 2.4)
    CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                         st->mv_bl, st->mv_vic_cuda_pl, nullptr));
    CHECK_VPI(vpiStreamSync(st->stream));

    // Lock pitch-linear MV as CUDA
    VPIImageData mvdata{};
    CHECK_VPI(vpiImageLockData(st->mv_vic_cuda_pl, VPI_LOCK_READ,
                              VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &mvdata));

    const int16_t *mvPtr = (const int16_t*)mvdata.buffer.pitch.planes[0].data;
    int mvPitchBytes     = (int)mvdata.buffer.pitch.planes[0].pitchBytes;

    const int mvW = (W + st->grid - 1) / st->grid;
    const int mvH = (H + st->grid - 1) / st->grid;

    // ROI in MV space derived from normalized settings.
    int x0 = (int)((float)mvW * st->roiX0 + 0.5f);
    int x1 = (int)((float)mvW * st->roiX1 + 0.5f);
    int y0 = (int)((float)mvH * st->roiY0 + 0.5f);
    int y1 = (int)((float)mvH * st->roiY1 + 0.5f);


    // Clamp and ensure non-empty ROI (at least a few cells).
    x0 = std::clamp(x0, 0, mvW - 1);
    x1 = std::clamp(x1, x0 + 1, mvW);
    y0 = std::clamp(y0, 0, mvH - 1);
    y1 = std::clamp(y1, y0 + 1, mvH);


    MVParams p{};
    p.mvW = mvW; p.mvH = mvH;
    p.mvPitchBytes = mvPitchBytes;
    p.grid = st->grid;

    p.x0 = x0; p.x1 = x1;
    p.y0 = y0; p.y1 = y1;
    p.step = st->overlayStep;

    p.minMag = st->mvMinMag;
    p.maxMag = st->mvMaxMag;

    p.minSamples = st->minSamples;
    p.cohMin     = st->cohMin;
    p.stdMax     = st->stdMax;
    p.tailRatio  = st->tailRatio;
    p.tailMinAbs = st->tailMinAbs;

    p.pxPerMeter = st->pxPerM;
    p.dtSec      = useDt;

    p.alphaHi = st->emaAlphaHi;
    p.alphaLo = st->emaAlphaLo;

    p.cohExcellent    = std::min(0.85f, st->cohMin + 0.30f);
    p.stdExcellentMax = st->stdMax * 0.60f;

    // GPU reduction + gating + EMA
    mv_reduce_gating_ema_cuda(mvPtr, st->d_ema, &p, st->d_speed_out, nullptr);

    // Copy results to CPU (tiny)
    cudaMemcpy(&st->speed_host, st->d_speed_out, sizeof(float), cudaMemcpyDeviceToHost);

    DevEmaState emaHost{};
    cudaMemcpy(&emaHost, st->d_ema, sizeof(DevEmaState), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    st->res_dx_host = emaHost.lastResDx;
    st->res_dy_host = emaHost.lastResDy;

    // Print on CPU
    print_speed_mps(st->speed_host);

    // Overlay on GPU
    if (st->overlayEnabled) {
        overlay_draw_mvs_nv12(image, W, H,
                              mvPtr, mvPitchBytes,
                              mvW, mvH, st->grid,
                              x0, x1, y0, y1,
                              st->overlayStep, st->overlayScale,
                              0.30f,
                              76, 85, 255,
                              st->res_dx_host, st->res_dy_host,
                              29, 255, 107);
    }

    CHECK_VPI(vpiImageUnlock(st->mv_vic_cuda_pl));

    // Advance
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
    // No frees here.
}

extern "C" void init(CustomerFunction *f)
{
    if (!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;
}

extern "C" void deinit(void)
{
    // No prints here.
}
