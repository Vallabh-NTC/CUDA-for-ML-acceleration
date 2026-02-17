// nvivafilter_vpi_of.cpp
//
// PURE DOF (Dense Optical Flow) + OVERLAY + GPU ANTI-SPIKE FILTER (NO QUALITY GATE)
// + Telemetry CSV integration (ax -> v) and visualize IMU velocity vector.
//
// NEW (requested):
// - Compute per-frame angle between the image horizontal axis (+X) and the DOF resultant vector.
// - Print that angle together with speed.
// - Save that angle into the runtime CSV.
//
// Pipeline:
// NVDEC -> NVMM(NV12 EGLImage) -> nvivafilter (this .so) -> NVMM
//
// - VPI 2.4.x + OFA Dense Optical Flow
// - Motion vector reduction on GPU
// - Anti-spike filtering on GPU (stateful) to reject huge speed jumps
// - (Optional) EMA smoothing on the filtered vector
//
// NOTE: All "quality metric" gating logic has been REMOVED as requested.
//
// Telemetry:
// - Telemetry CSV (VPI_OF_TELEM_CSV) containing unix_sec/unix_nsec + ax (+ optional v_corrected or v)
// - Integrate ax -> v_mps (optionally lightly anchored to v_corrected to limit drift)
// - Draw IMU velocity vector with direction from DOF resultant and magnitude from telemetry.
//

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath>
#include <mutex>
#include <string>

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
#include "mv_reduce.hpp"
#include "overlay.hpp"
#include "mv_spike_filter.hpp"

// ----------------------------
// VPI error helper
// ----------------------------
#define CHECK_VPI(stmt)                                                         \
    do {                                                                        \
        VPIStatus _st = (stmt);                                                 \
        if (_st != VPI_SUCCESS) {                                               \
            char msg[512];                                                      \
            vpiGetLastStatusMessage(msg, sizeof(msg));                          \
            std::fprintf(stderr, "[vpi_of_pure] VPI error: %s at %s\n",         \
                         vpiStatusGetName(_st), #stmt);                         \
            std::fprintf(stderr, "[vpi_of_pure] Message: %s\n", msg);           \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// ----------------------------
// Env helpers
// ----------------------------
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

// ----------------------------
// Small math helpers
// ----------------------------
static inline float rad2deg(float r) { return r * (180.0f / 3.14159265358979323846f); }

// Signed included angle between the horizontal *line* and the DOF resultant vector.
// Left/right are equivalent.
// Sign convention (requested):
//   dy < 0  => positive angle  (vector points DOWN)
//   dy > 0  => negative angle  (vector points UP)
// Range: [-90 .. +90] degrees.
static inline float dof_angle_deg(float dx, float dy)
{
    // If vector is ~zero, return 0.
    if ((std::fabs(dx) + std::fabs(dy)) < 1e-9f) return 0.0f;

    // Fold left/right using |dx|
    float a = rad2deg(std::atan2(dy, std::fabs(dx)));

    // Invert sign (THIS is the only change)
    a = -a;

    // Safety clamp
    if (a >  90.0f) a =  90.0f;
    if (a < -90.0f) a = -90.0f;

    return a;
}




// ----------------------------
// Tiny CSV helper
// ----------------------------
static std::vector<std::string> csv_split_line(const char *s)
{
    std::vector<std::string> out;
    std::string cur;
    for (const char *p = s; ; ++p) {
        char c = *p;
        if (c == ',' || c == '\n' || c == '\r' || c == 0) {
            out.push_back(cur);
            cur.clear();
            if (c == 0 || c == '\n' || c == '\r') break;
        } else {
            cur.push_back(c);
        }
    }
    return out;
}

static bool str_isfinite_float(const char *s)
{
    if (!s || !*s) return false;
    // crude: rely on atof + isfinite
    float v = std::atof(s);
    return std::isfinite(v);
}

// ----------------------------
// State
// ----------------------------
struct State
{
    bool inited = false;
    int W = 0, H = 0;

    VPIStream stream = nullptr;
    float steerRatio = 1.0f;   // conversion DOF_angle -> steering angle


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

    // OFA output MV in block-linear + pitch-linear for CUDA read
    VPIImage mv_bl = nullptr;
    VPIImage mv_vic_cuda_pl = nullptr;

    bool havePrev = false;

    // ROI in normalized coordinates [0..1] in MV space
    float roiX0 = 0.15f, roiX1 = 0.85f;
    float roiY0 = 0.20f, roiY1 = 0.80f;

    // Sampling step in MV cells
    int step = 1;

    // Calibration (pixels per meter)
    float pxPerM = 717.0f;

    // dt policy:
    // - If forceFps > 0, dtSec = 1/forceFps (deterministic)
    // - else dtSec measured from CPU clock
    int forceFps = 15; // set 0 to use measured dt

    // Timing
    bool haveTime = false;
    std::chrono::steady_clock::time_point lastTime;

    // Pure reducer output on GPU + host copy
    MVPureOut *d_pure = nullptr;
    MVPureOut  pure_h{};

    // ----------------------------
    // CSV logging
    // ----------------------------
    bool  csvEnabled = false;
    int   csvEveryN  = 1;             // write every N frames (1 = every frame)
    FILE *csvFp      = nullptr;

    bool  csvHaveT0 = false;
    std::chrono::steady_clock::time_point csvT0;

    std::mutex csvMutex;

    // ----------------------------
    // GPU anti-spike (stateful)
    // ----------------------------
    MVSpikeFilterState *d_spikeState = nullptr;
    MVPureOut          *d_pure_filt  = nullptr;

    // Anti-spike knobs:
    // Treat jump >= spikeKmh as spike unless it persists stableFrames frames.
    float spikeKmh = 10.0f;
    float okKmh    = 2.0f;   // kept for env compatibility; kernel currently doesn't use it
    int   stableFrames = 80;

    // Overlay controls
    bool  overlayEnabled = true;
    float overlayScale   = 3.5f;
    float minMagDraw     = 0.30f;

    uint32_t frameId = 0;

    // ----------------------------
    // Low-pass filter (EMA)
    // ----------------------------
    bool  emaHave = false;
    float ema_dx = 0.0f;
    float ema_dy = 0.0f;
    float emaTauSec = 0.25f;   // time constant in seconds

    // ----------------------------
    // Telemetry CSV integration
    // ----------------------------
    bool telemEnabled = false;
    int  telemFrameOffset = 0;
    float telemAnchorK = 0.05f;              // 0..1
    std::vector<double> telemTsec;           // unix timestamp (sec)
    std::vector<float>  telemSpeedMps;       // integrated speed (m/s)
};

static State* get_or_create_state(void **userPtr)
{
    if (userPtr && *userPtr) return reinterpret_cast<State*>(*userPtr);
    State *st = new State();
    if (userPtr) *userPtr = st;
    return st;
}

// ----------------------------
// Telemetry loader (ax -> v)
// ----------------------------
static void load_telemetry_if_any(State *st)
{
    const char *path = std::getenv("VPI_OF_TELEM_CSV");
    if (!path || !path[0]) return;

    st->telemFrameOffset = get_env_int("VPI_OF_TELEM_FRAME_OFFSET", 0);
    st->telemAnchorK     = std::clamp(get_env_float("VPI_OF_TELEM_ANCHOR_K", st->telemAnchorK), 0.0f, 1.0f);

    FILE *fp = std::fopen(path, "r");
    if (!fp) {
        std::fprintf(stderr, "[vpi_of_pure] ERROR: cannot open telemetry CSV: %s\n", path);
        return;
    }

    char line[4096];
    if (!std::fgets(line, sizeof(line), fp)) { std::fclose(fp); return; }

    std::vector<std::string> hdr = csv_split_line(line);

    auto col = [&](const char *name)->int {
        for (int i = 0; i < (int)hdr.size(); ++i) {
            if (hdr[i] == name) return i;
        }
        return -1;
    };

    int c_sec  = col("unix_sec");
    int c_nsec = col("unix_nsec");
    int c_ax   = col("ax");
    int c_vcor = col("v_corrected");
    int c_v    = col("v");

    if (c_sec < 0 || c_nsec < 0 || c_ax < 0) {
        std::fprintf(stderr, "[vpi_of_pure] ERROR: telemetry CSV missing unix_sec/unix_nsec/ax\n");
        std::fclose(fp);
        return;
    }

    std::vector<double> tsec;
    std::vector<float>  ax;
    std::vector<float>  vref_kmh;

    while (std::fgets(line, sizeof(line), fp)) {
        std::vector<std::string> f = csv_split_line(line);
        if ((int)f.size() < (int)hdr.size()) continue;

        double sec  = std::atof(f[c_sec].c_str());
        double nsec = std::atof(f[c_nsec].c_str());
        double ts   = sec + nsec * 1e-9;

        float axv = std::atof(f[c_ax].c_str());

        float vck = NAN;
        if (c_vcor >= 0 && str_isfinite_float(f[c_vcor].c_str())) {
            vck = std::atof(f[c_vcor].c_str());
        } else if (c_v >= 0 && str_isfinite_float(f[c_v].c_str())) {
            vck = std::atof(f[c_v].c_str());
        }

        tsec.push_back(ts);
        ax.push_back(axv);
        vref_kmh.push_back(vck);
    }

    std::fclose(fp);

    if (tsec.size() < 2) {
        std::fprintf(stderr, "[vpi_of_pure] Telemetry CSV too short\n");
        return;
    }

    std::vector<float> v_mps(tsec.size(), 0.0f);

    // init v0 from first available reference speed if present
    for (size_t i = 0; i < tsec.size(); ++i) {
        if (std::isfinite(vref_kmh[i])) {
            v_mps[0] = vref_kmh[i] * (1000.0f / 3600.0f);
            break;
        }
    }

    for (size_t i = 1; i < tsec.size(); ++i) {
        double dt = tsec[i] - tsec[i - 1];
        if (!(dt > 0.0 && dt < 0.5)) dt = 1.0 / 50.0; // safe clamp
        v_mps[i] = v_mps[i - 1] + ax[i] * (float)dt;

        // Optional anchoring to external speed to reduce drift
        if (st->telemAnchorK > 0.0f && std::isfinite(vref_kmh[i])) {
            float vref = vref_kmh[i] * (1000.0f / 3600.0f);
            v_mps[i] = (1.0f - st->telemAnchorK) * v_mps[i] + st->telemAnchorK * vref;
        }
    }

    st->telemEnabled = true;
    st->telemTsec = std::move(tsec);
    st->telemSpeedMps = std::move(v_mps);

    std::fprintf(stderr,
                 "[vpi_of_pure] Telemetry loaded: %zu samples from %s (offset=%d, anchorK=%.3f)\n",
                 st->telemSpeedMps.size(), path, st->telemFrameOffset, st->telemAnchorK);
}

// ----------------------------
// Init VPI
// ----------------------------
static void init_vpi_if_needed(State *st, int W, int H)
{
    if (st->inited) return;

    st->W = W; st->H = H;

    // Runtime knobs
    st->roiX0    = std::clamp(get_env_float("VPI_OF_ROI_X0", st->roiX0), 0.0f, 1.0f);
    st->roiX1    = std::clamp(get_env_float("VPI_OF_ROI_X1", st->roiX1), 0.0f, 1.0f);
    st->roiY0    = std::clamp(get_env_float("VPI_OF_ROI_Y0", st->roiY0), 0.0f, 1.0f);
    st->roiY1    = std::clamp(get_env_float("VPI_OF_ROI_Y1", st->roiY1), 0.0f, 1.0f);
    st->step     = std::max(1, get_env_int("VPI_OF_STEP", st->step));
    st->pxPerM   = std::max(1.0f, get_env_float("VPI_OF_PX_PER_M", st->pxPerM));
    st->forceFps = get_env_int("VPI_OF_FORCE_FPS", st->forceFps); // 0 = use measured dt

    st->steerRatio = get_env_float("VPI_OF_STEER_RATIO", 1.0f);


    // Overlay knobs
    st->overlayEnabled = (get_env_int("VPI_OF_OVERLAY", 1) != 0);
    st->overlayScale   = std::max(0.1f, get_env_float("VPI_OF_OVERLAY_SCALE", st->overlayScale));
    st->minMagDraw     = std::max(0.0f, get_env_float("VPI_OF_MIN_MAG_DRAW", st->minMagDraw));

    // Anti-spike knobs (ONLY)
    st->spikeKmh     = std::max(0.0f, get_env_float("VPI_OF_SPIKE_KMH", st->spikeKmh));
    st->okKmh        = std::max(0.0f, get_env_float("VPI_OF_OK_KMH", st->okKmh));
    st->stableFrames = std::max(1,    get_env_int  ("VPI_OF_STABLE_FRAMES", st->stableFrames));

    // EMA knob
    st->emaTauSec = std::max(0.0f, get_env_float("VPI_OF_EMA_TAU", st->emaTauSec));

    // Sanity ROI
    if (!(st->roiX1 > st->roiX0 + 0.01f)) { st->roiX0 = 0.15f; st->roiX1 = 0.85f; }
    if (!(st->roiY1 > st->roiY0 + 0.01f)) { st->roiY0 = 0.20f; st->roiY1 = 0.80f; }

    // CSV knobs
    st->csvEnabled = (get_env_int("VPI_OF_CSV", 0) != 0);
    st->csvEveryN  = std::max(1, get_env_int("VPI_OF_CSV_EVERY", 1));

    if (st->csvEnabled && !st->csvFp) {
        const char *path = std::getenv("VPI_OF_CSV_PATH");
        if (!path) path = "/tmp/vpi_of_speed.csv";

        st->csvFp = std::fopen(path, "w");
        if (st->csvFp) {
            // NEW: angle_deg
            std::fprintf(st->csvFp,
                "t_sec,speed_mps,speed_kmh,angle_deg,dof_steer,dx_pxpf,dy_pxpf,count,dt_sec\n");
            std::fflush(st->csvFp);
        } else {
            std::fprintf(stderr, "[vpi_of_pure] ERROR: cannot open CSV at %s\n", path);
            st->csvEnabled = false;
        }
    }

    // Telemetry (optional)
    load_telemetry_if_any(st);

    CHECK_VPI(vpiStreamCreate(0, &st->stream));

    // CUDA-backed Y8 images
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &st->prev_y8_pl));
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &st->cur_y8_pl));

    st->numLevels = max_levels_scale_half_min32(W, H);

    // Pitch-linear pyramids (CUDA)
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER,
                              st->numLevels, st->pyrScale, 0, &st->prev_pyr_pl));
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER,
                              st->numLevels, st->pyrScale, 0, &st->cur_pyr_pl));

    // Block-linear pyramids (VIC->OFA)
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

    // BL -> pitch-linear via VIC, and readable by CUDA
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16,
                             VPI_BACKEND_VIC | VPI_BACKEND_CUDA, &st->mv_vic_cuda_pl));

    // Pure reducer output on GPU
    cudaMalloc(&st->d_pure, sizeof(MVPureOut));
    cudaMemset(st->d_pure, 0, sizeof(MVPureOut));

    // GPU anti-spike state + filtered output
    cudaMalloc(&st->d_spikeState, sizeof(MVSpikeFilterState));
    cudaMemset(st->d_spikeState, 0, sizeof(MVSpikeFilterState));

    cudaMalloc(&st->d_pure_filt, sizeof(MVPureOut));
    cudaMemset(st->d_pure_filt, 0, sizeof(MVPureOut));

    st->inited = true;
}

// ----------------------------
// Main GPU process
// ----------------------------
static void gpu_process(EGLImageKHR image, void **userPtr)
{
    State *st = get_or_create_state(userPtr);

    static bool cu_inited = false;
    if (!cu_inited) { cuInit(0); cu_inited = true; }

    int W = st->W, H = st->H;
    if (W <= 0 || H <= 0) {
        if (!get_size_from_env(W, H)) return;
        st->W = W; st->H = H;
    }

    init_vpi_if_needed(st, W, H);

    // Compute dtSec
    float useDt = 1.0f / 15.0f;
    if (st->forceFps > 0) {
        useDt = 1.0f / (float)st->forceFps;
    } else {
        auto now = std::chrono::steady_clock::now();
        float dtSec = 0.0f;
        if (!st->haveTime) {
            st->haveTime = true;
            st->lastTime = now;
        } else {
            dtSec = std::chrono::duration<float>(now - st->lastTime).count();
            st->lastTime = now;
        }
        useDt = (dtSec > 0.0f) ? std::clamp(dtSec, 1.0f/120.0f, 1.0f/5.0f) : (1.0f/15.0f);
    }

    // Copy NV12 luma (Y) from EGLImage into CUDA-backed VPI Y8
    VPIImageData ydata{};
    CHECK_VPI(vpiImageLockData(st->cur_y8_pl, VPI_LOCK_WRITE,
                              VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &ydata));

    uint8_t *dstY = (uint8_t*)ydata.buffer.pitch.planes[0].data;
    int dstPitch  = (int)ydata.buffer.pitch.planes[0].pitchBytes;

    egl_nv12_copy_y_to_cuda(image, dstY, dstPitch, W, H);

    CHECK_VPI(vpiImageUnlock(st->cur_y8_pl));

    // First frame: no flow.
    if (!st->havePrev) {
        std::swap(st->prev_y8_pl, st->cur_y8_pl);
        st->havePrev = true;
        st->frameId++;
        return;
    }

    // Phase A: CUDA pyramids
    CHECK_VPI(vpiSubmitGaussianPyramidGenerator(st->stream, VPI_BACKEND_CUDA,
                                               st->prev_y8_pl, st->prev_pyr_pl, VPI_BORDER_CLAMP));
    CHECK_VPI(vpiSubmitGaussianPyramidGenerator(st->stream, VPI_BACKEND_CUDA,
                                               st->cur_y8_pl,  st->cur_pyr_pl,  VPI_BORDER_CLAMP));
    CHECK_VPI(vpiStreamSync(st->stream));

    // Phase B: VIC PL->BL for OFA
    for (int lvl = 0; lvl < st->numLevels; ++lvl) {
        CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                             st->prevLvlPL[lvl], st->prevLvlBL[lvl], nullptr));
        CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                             st->curLvlPL[lvl],  st->curLvlBL[lvl],  nullptr));
    }
    CHECK_VPI(vpiStreamSync(st->stream));

    // Phase C: OFA dense optical flow
    CHECK_VPI(vpiSubmitOpticalFlowDensePyramid(st->stream, VPI_BACKEND_OFA,
                                              st->ofa_payload,
                                              st->prev_pyr_bl, st->cur_pyr_bl,
                                              st->mv_bl));
    CHECK_VPI(vpiStreamSync(st->stream));

    // Phase D: BL -> pitch-linear MV via VIC
    CHECK_VPI(vpiSubmitConvertImageFormat(st->stream, VPI_BACKEND_VIC,
                                         st->mv_bl, st->mv_vic_cuda_pl, nullptr));
    CHECK_VPI(vpiStreamSync(st->stream));

    // Lock MV as CUDA pitch-linear
    VPIImageData mvdata{};
    CHECK_VPI(vpiImageLockData(st->mv_vic_cuda_pl, VPI_LOCK_READ,
                              VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &mvdata));

    const int16_t *mvPtr = (const int16_t*)mvdata.buffer.pitch.planes[0].data;
    int mvPitchBytes     = (int)mvdata.buffer.pitch.planes[0].pitchBytes;

    const int mvW = (W + st->grid - 1) / st->grid;
    const int mvH = (H + st->grid - 1) / st->grid;

    // ROI in MV space
    int x0 = (int)((float)mvW * st->roiX0 + 0.5f);
    int x1 = (int)((float)mvW * st->roiX1 + 0.5f);
    int y0 = (int)((float)mvH * st->roiY0 + 0.5f);
    int y1 = (int)((float)mvH * st->roiY1 + 0.5f);

    x0 = std::clamp(x0, 0, mvW - 1);
    x1 = std::clamp(x1, x0 + 1, mvW);
    y0 = std::clamp(y0, 0, mvH - 1);
    y1 = std::clamp(y1, y0 + 1, mvH);

    // Pure reduction params
    MVPureParams p{};
    p.mvW = mvW; p.mvH = mvH;
    p.mvPitchBytes = mvPitchBytes;
    p.grid = st->grid;
    p.x0 = x0; p.x1 = x1;
    p.y0 = y0; p.y1 = y1;
    p.step = st->step;
    p.pxPerMeter = st->pxPerM;
    p.dtSec = useDt;

    // 1) Raw reduction on GPU
    mv_reduce_pure_cuda(mvPtr, &p, st->d_pure);

    // 2) Anti-spike on GPU (stateful)
    MVSpikeFilterParams sfp{};
    sfp.ok_kmh        = st->okKmh;
    sfp.spike_kmh     = st->spikeKmh;
    sfp.stable_frames = st->stableFrames;

    mv_spike_filter_cuda(st->d_pure, &sfp, st->d_spikeState, st->d_pure_filt);

    // Copy FILTERED result to host
    MVPureOut filt_h{};
    cudaMemcpy(&filt_h, st->d_pure_filt, sizeof(MVPureOut), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Base output = filtered
    st->pure_h = filt_h;

    // Optional EMA on the filtered vector (do NOT overwrite after)
    if (st->emaTauSec > 0.0f) {
        float alpha = useDt / (st->emaTauSec + useDt);

        if (!st->emaHave) {
            st->emaHave = true;
            st->ema_dx = st->pure_h.mean_dx;
            st->ema_dy = st->pure_h.mean_dy;
        } else {
            st->ema_dx += alpha * (st->pure_h.mean_dx - st->ema_dx);
            st->ema_dy += alpha * (st->pure_h.mean_dy - st->ema_dy);
        }

        st->pure_h.mean_dx = st->ema_dx;
        st->pure_h.mean_dy = st->ema_dy;
        st->pure_h.res_mag = std::sqrt(st->ema_dx*st->ema_dx + st->ema_dy*st->ema_dy);
        st->pure_h.speed_mps = (st->pure_h.res_mag / st->pxPerM) / useDt;
    }

    // NEW: compute DOF angle w.r.t. horizontal axis (+X)
    float angle_deg = dof_angle_deg(st->pure_h.mean_dx, st->pure_h.mean_dy);
    // Convert DOF angle into steering estimate using calibration ratio
    float dof_steer = angle_deg * st->steerRatio;


    // ---------------------------------------------------------
    // Telemetry vector (IMU) in px/frame:
    // direction = DOF resultant, magnitude = integrated v (m/s)
    // ---------------------------------------------------------
    float imuDxPx = 0.0f, imuDyPx = 0.0f;
    float imuScale = 5.0f; // we pass px/frame already

    if (st->telemEnabled && !st->telemSpeedMps.empty()) {
        int ti = (int)st->frameId + st->telemFrameOffset;
        if (ti >= 0 && ti < (int)st->telemSpeedMps.size()) {
            float v_mps = st->telemSpeedMps[ti];

            float rx = st->pure_h.mean_dx;
            float ry = st->pure_h.mean_dy;
            float r2 = rx*rx + ry*ry;

            if (r2 > 1e-8f) {
                float rinv = 1.0f / std::sqrt(r2);
                float dirx = rx * rinv;
                float diry = ry * rinv;

                // m/s -> px/frame
                float v_pxpf = v_mps * st->pxPerM * useDt;

                imuDxPx = dirx * v_pxpf;
                imuDyPx = diry * v_pxpf;
            }
        }
    }

    // Write CSV (filtered/EMA values only)
    if (st->csvEnabled && st->csvFp && (st->frameId % (uint32_t)st->csvEveryN == 0)) {
        std::lock_guard<std::mutex> lk(st->csvMutex);

        auto now = std::chrono::steady_clock::now();
        if (!st->csvHaveT0) {
            st->csvHaveT0 = true;
            st->csvT0 = now;
        }
        double t_sec = std::chrono::duration<double>(now - st->csvT0).count();
        double kmh = (double)st->pure_h.speed_mps * 3.6;

        // NEW: angle_deg column
        std::fprintf(st->csvFp,
                     "%.6f,%.6f,%.6f,%.3f,%.3f,%.6f,%.6f,%d,%.6f\n",
                     t_sec,
                     (double)st->pure_h.speed_mps,
                     kmh,
                     (double)angle_deg,
                     dof_steer,
                     (double)st->pure_h.mean_dx,
                     (double)st->pure_h.mean_dy,
                     st->pure_h.count,
                     (double)useDt);

        if ((st->frameId % 120) == 0) {
            std::fflush(st->csvFp);
        }
    }

    // Debug line (NEW: angle)
    std::fprintf(stdout,
                "speed=%.3f m/s (%.2f km/h)  angle=%.2f deg DOF_steer=%.2f  res=(%.3f,%.3f) px/frame  n=%d  dt=%.4f  telem=(%.3f,%.3f)\n",
                st->pure_h.speed_mps,
                st->pure_h.speed_mps * 3.6f,
                angle_deg,
                dof_steer,
                st->pure_h.mean_dx,
                st->pure_h.mean_dy,
                st->pure_h.count,
                useDt,
                imuDxPx, imuDyPx);
    std::fflush(stdout);

    // OVERLAY: use filtered/EMA resultant + telemetry IMU vector
    if (st->overlayEnabled) {
        overlay_draw_mvs_nv12(
            image, W, H,
            mvPtr, mvPitchBytes,
            mvW, mvH, st->grid,
            x0, x1, y0, y1,
            st->step, st->overlayScale,
            st->minMagDraw,
            76, 85, 255,                        // field color (YUV)
            st->pure_h.mean_dx, st->pure_h.mean_dy,
            29, 255, 107,                       // OF resultant color (YUV)
            // IMU / Telemetry vector (px/frame)
            imuDxPx, imuDyPx,
            imuScale,
            145, 54, 34,                        // IMU color (YUV) (tweak if needed)
            0.0f, 1e9f,                         // forceDeg=0 disables forcing
            st->frameId);
    }

    CHECK_VPI(vpiImageUnlock(st->mv_vic_cuda_pl));

    // Advance
    std::swap(st->prev_y8_pl, st->cur_y8_pl);
    st->frameId++;
}

// ----------------------------
// nvivafilter hooks
// ----------------------------
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
                         ColorFormat*, unsigned int, void **userPtr)
{
    State *st = (userPtr && *userPtr) ? reinterpret_cast<State*>(*userPtr) : nullptr;
    if (!st) return;

    if (st->csvFp) {
        std::fflush(st->csvFp);
        std::fclose(st->csvFp);
        st->csvFp = nullptr;
    }

    // NOTE: you can free VPI/CUDA resources here if you want.
    // Keeping minimal to match your existing behavior.
}

extern "C" void init(CustomerFunction *f)
{
    if (!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;
}

extern "C" void deinit(void) {}
