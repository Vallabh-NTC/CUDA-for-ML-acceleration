// nvivafilter_raft.cpp — RAFT zero-copy optical flow pipeline
// Estimates longitudinal (vx) and lateral (vy) vehicle speed in km/h
// from raw camera frames decoded by NVDEC on Jetson Orin AGX.
//
// Processing chain:
//   EGLImage (NVMM) → NV12→float32 → Sharpen → RAFT TRT → flow_reduce
//   → axis mapping (TILTED mount) → FOE correction → Startup filter → EMA
//   → CSV + stdout → overlay
//
// ── AXIS MAPPING (tilted mount, 2026-06) ───────────────────────────────────
//   The camera is no longer parallel to the ground. With this mounting the
//   vehicle's FORWARD motion appears as VERTICAL image flow (mean_v), and the
//   lateral / yaw motion appears as HORIZONTAL image flow (mean_u). This is
//   the opposite of the old parallel mount, where forward motion was in u.
//
//     vx (forward) = -mean_v * scale       (was -mean_u)
//     vy (lateral) =  mean_u * scale        (was  mean_v)
//
//   Validated against vehicle telemetry (eth_VDSO_Vx3dKmph):
//     corr(-mean_v, Vx) = 0.98, error mean -0.07 km/h, std 0.68 km/h (0–14 km/h).
//
//   NOTE: vy now derives from mean_u, which also picks up yaw rotation at low
//   speed — vy / beta still need a yaw-decoupling step before they are trusted.
//
// The FOE correction now applies to the LATERAL channel (mean_u), scaled by
// the forward signal (mean_v):
//     lat_corrected = mean_u - (foe_a * mean_v + foe_b)
// With FOE_A=FOE_B=0 (default) it is a no-op. foe_correct_flow() in overlay.cu
// still corrects the v-channel of the flow field for visualization only; it is
// also a no-op while FOE is zero. Revisit it when recalibrating FOE on u.
//
// Speed scale (px/frame → km/h):
//     scale = (1 / PX_PER_M) * FPS * 3.6
//   FPS is now configurable via RAFT_FPS (default 30) — previously hardcoded
//   to 100 for the old 100-fps camera. PX_PER_M default updated to 55.
//
// Startup filter (active until first stable lock):
//   Gate 1 — absolute bounds (vx >= MIN_VX, |vy| <= MAX_VY)
//   Gate 2 — gap-aware derivative bound (|dvx| <= MAX_DVX)
//   Gate 3 — consecutive valid frames (MIN_CONSECUTIVE in a row)
//   Once locked, all frames are emitted unconditionally.
//   Set the bounds wide (e.g. MIN_VX very negative) to effectively disable.
//
// EMA filter (active after lock, initialized at lock value — no transient):
//   vx: alpha = RAFT_ALPHA_VX (default 0.4)
//   vy: alpha = RAFT_ALPHA_VY (default 0.2)
//   Set RAFT_ALPHA_VX=1.0 / RAFT_ALPHA_VY=1.0 to disable.
//
// Locale note: setlocale(LC_NUMERIC, "C") is forced at init so that atof()
// parses env vars with '.' as decimal separator, regardless of the system
// locale (e.g. it_IT.UTF-8 would otherwise parse "0.62" as 0.0).

#include "egl_map.hpp"
#include "nv12_to_rgb_fp16.hpp"
#include "preprocess.hpp"
#include "raft_infer.hpp"
#include "flow_reduce.hpp"
#include "overlay.hpp"
#include "nvivafilter_customer_api.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <clocale>

static int         env_int  (const char *k, int   d) { const char *v=getenv(k); return v?atoi(v):d; }
static float       env_float(const char *k, float d) { const char *v=getenv(k); return v?atof(v):d; }
static const char *env_str  (const char *k, const char *d) { const char *v=getenv(k); return v?v:d; }


// ── EMA filter ────────────────────────────────────────────────────────────────
struct EMA {
    float alpha  = 1.0f;
    float value  = 0.0f;
    bool  inited = false;

    void  init(float a) { alpha = a; }

    float push(float x) {
        if (!inited) { value = x; inited = true; return x; }
        value = alpha * x + (1.0f - alpha) * value;
        return value;
    }
};


// ── Pipeline State ────────────────────────────────────────────────────────────
struct State {
    bool  inited      = false;
    bool  init_failed = false;
    int   W = 0, H = 0;

    // ROI in normalized coordinates [0,1]
    float roi_x0 = 0.15f;
    float roi_x1 = 0.85f;
    float roi_y0 = 0.25f;
    float roi_y1 = 0.80f;

    // Overlay parameters
    int   step           = 16;
    float arrow_scale    = 4.0f;
    float min_mag        = 1.5f;
    float result_scale   = 8.0f;
    float sharp_strength = 1.5f;

    // Speed conversion (px/frame → km/h)
    // SCALE = (1 / px_per_m) * fps * 3.6
    float scale_kmh = 0.0f;

    // Startup gate 1 — absolute bounds
    float min_vx_kmh = 5.0f;
    float max_vy_kmh = 3.0f;

    // Startup gate 2 — gap-aware derivative bound
    float    max_dvx_kmh      = 5.0f;
    int      max_valid_gap    = 3;
    float    last_valid_vx    = -1.0f;
    uint32_t last_valid_frame = 0;

    // Startup gate 3 — consecutive valid frames
    int  min_consecutive   = 3;
    int  consecutive_valid = 0;

    // Lock flag
    bool locked = false;

    // EMA filters
    EMA  ema_vx;
    EMA  ema_vy;

    RaftInfer    raft;
    float       *d_frame_prev = nullptr;
    float       *d_frame_curr = nullptr;
    float       *d_flow       = nullptr;
    FlowResult  *d_result     = nullptr;

    bool         have_prev = false;
    cudaStream_t stream    = nullptr;
    uint32_t     frame_id  = 0;

    FILE        *csv_file = nullptr;

    // FOE correction coefficients (now applied to the lateral channel, mean_u)
    // lat_corrected = mean_u - (foe_a * mean_v + foe_b)
    float foe_a = 0.0f;
    float foe_b = 0.0f;
};

static State *get_state(void **p) {
    if (p && *p) return reinterpret_cast<State*>(*p);
    auto *st = new State();
    if (p) *p = st;
    return st;
}

static bool init_once(State *st, int W, int H)
{
    if (st->inited)      return true;
    if (st->init_failed) return false;

    // Force C locale for numeric parsing — makes atof() use '.' as decimal
    // separator regardless of the system locale (e.g. it_IT.UTF-8).
    std::setlocale(LC_NUMERIC, "C");

    st->W = W; st->H = H;

    st->roi_x0          = env_float("RAFT_ROI_X0",          st->roi_x0);
    st->roi_x1          = env_float("RAFT_ROI_X1",          st->roi_x1);
    st->roi_y0          = env_float("RAFT_ROI_Y0",          st->roi_y0);
    st->roi_y1          = env_float("RAFT_ROI_Y1",          st->roi_y1);
    st->step            = env_int  ("RAFT_STEP",             st->step);
    st->arrow_scale     = env_float("RAFT_ARROW_SCALE",      st->arrow_scale);
    st->min_mag         = env_float("RAFT_MIN_MAG",          st->min_mag);
    st->result_scale    = env_float("RAFT_RESULT_SCALE",     st->result_scale);
    st->sharp_strength  = env_float("RAFT_SHARP",            st->sharp_strength);
    st->min_vx_kmh      = env_float("RAFT_MIN_VX_KMH",      st->min_vx_kmh);
    st->max_vy_kmh      = env_float("RAFT_MAX_VY_KMH",      st->max_vy_kmh);
    st->max_dvx_kmh     = env_float("RAFT_MAX_DVX_KMH",     st->max_dvx_kmh);
    st->max_valid_gap   = env_int  ("RAFT_MAX_VALID_GAP",   st->max_valid_gap);
    st->min_consecutive = env_int  ("RAFT_MIN_CONSECUTIVE", st->min_consecutive);

    st->ema_vx.init(env_float("RAFT_ALPHA_VX", 0.4f));
    st->ema_vy.init(env_float("RAFT_ALPHA_VY", 0.2f));

    st->foe_a = env_float("RAFT_FOE_A", 0.0f);
    st->foe_b = env_float("RAFT_FOE_B", 0.0f);

    // Speed scale — read once at init, not per frame.
    // FPS is configurable now (was hardcoded 100 for the old camera).
    const float px_per_m = env_float("RAFT_PX_PER_M", 55.0f);
    const float fps      = env_float("RAFT_FPS",       30.0f);
    st->scale_kmh = (1.0f / px_per_m) * fps * 3.6f;

    const char *engine_path = env_str("RAFT_ENGINE_PATH",
        "/home/jetson-ntc/raft/raft_large_fp16.engine");

    const int rx0 = (int)(st->roi_x0 * W), rx1 = (int)(st->roi_x1 * W);
    const int ry0 = (int)(st->roi_y0 * H), ry1 = (int)(st->roi_y1 * H);
    int nx = 0, ny = 0;
    for (int x = rx0; x < rx1; x += st->step) nx++;
    for (int y = ry0; y < ry1; y += st->step) ny++;

    std::fprintf(stderr,
        "[raft_of] Init W=%d H=%d engine=%s\n"
        "[raft_of] AXIS MAP: forward = -mean_v, lateral = mean_u (TILTED mount)\n"
        "[raft_of] ROI px=[%d,%d]x[%d,%d]  step=%d  N=%d points\n"
        "[raft_of] FOE correction (on lateral)  : A=%.4f  B=%.3f\n"
        "[raft_of] startup gate 1 (absolute)   : vx >= %.1f km/h, |vy| <= %.1f km/h\n"
        "[raft_of] startup gate 2 (derivative) : |dvx| <= %.1f km/h/frame, gap <= %d frames\n"
        "[raft_of] startup gate 3 (consecutive): %d valid frames in a row to lock\n"
        "[raft_of] EMA filter                  : alpha_vx=%.2f  alpha_vy=%.2f\n"
        "[raft_of] Speed scale                 : px_per_m=%.1f fps=%.1f → %.6f (km/h per px/frame)\n",
        W, H, engine_path,
        rx0, rx1, ry0, ry1, st->step, nx * ny,
        st->foe_a, st->foe_b,
        st->min_vx_kmh, st->max_vy_kmh,
        st->max_dvx_kmh, st->max_valid_gap,
        st->min_consecutive,
        st->ema_vx.alpha, st->ema_vy.alpha,
        px_per_m, fps, st->scale_kmh);

    if (cudaStreamCreateWithFlags(&st->stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "[raft_of] cudaStreamCreate failed\n");
        st->init_failed = true; return false;
    }

    if (!st->raft.init(engine_path, W, H)) {
        std::fprintf(stderr, "[raft_of] RAFT init failed\n");
        st->init_failed = true; return false;
    }

    const size_t frame_bytes = sizeof(float) * 3 * H * W;
    const size_t flow_bytes  = sizeof(float) * 2 * H * W;

    if (cudaMalloc(&st->d_frame_prev, frame_bytes) != cudaSuccess ||
        cudaMalloc(&st->d_frame_curr, frame_bytes) != cudaSuccess ||
        cudaMalloc(&st->d_flow,       flow_bytes)  != cudaSuccess ||
        cudaMalloc(&st->d_result, sizeof(FlowResult)) != cudaSuccess) {
        std::fprintf(stderr, "[raft_of] cudaMalloc failed\n");
        st->init_failed = true; return false;
    }

    cudaMemset(st->d_frame_prev, 0, frame_bytes);
    cudaMemset(st->d_frame_curr, 0, frame_bytes);
    cudaMemset(st->d_flow,       0, flow_bytes);
    cudaMemset(st->d_result,     0, sizeof(FlowResult));

    const char *csv_path = env_str("RAFT_CSV_PATH", "/home/jetson-ntc/raft/output.csv");
    st->csv_file = std::fopen(csv_path, "w");
    if (!st->csv_file) {
        std::fprintf(stderr, "[raft_of] Cannot open CSV: %s\n", csv_path);
        st->init_failed = true; return false;
    }
    std::fprintf(st->csv_file, "frame,mean_u_px,mean_v_px,vx_kmh,vy_kmh\n");
    std::fflush(st->csv_file);

    st->inited = true;
    std::fprintf(stderr, "[raft_of] Init complete — CSV: %s\n", csv_path);
    return true;
}

static void gpu_process(EGLImageKHR image, void **userPtr)
{
    State *st = get_state(userPtr);

    int W = st->W > 0 ? st->W : env_int("RAFT_W", 672);
    int H = st->H > 0 ? st->H : env_int("RAFT_H", 376);

    if (!init_once(st, W, H)) return;

    // Step 1: map EGLImage → CUDA pointer (zero-copy)
    EGLMapResult egl;
    if (!egl_map(image, W, H, egl)) {
        std::fprintf(stderr, "[raft_of] egl_map failed frame %u\n", st->frame_id);
        return;
    }

    // Step 2: NV12 → float32 RGB [1,3,H,W]
    nv12_to_rgb_fp16(egl.d_y, egl.d_uv,
                     egl.pitchY, egl.pitchUV,
                     W, H, st->d_frame_curr, st->stream);

    // Step 3: unsharp mask sharpening
    preprocess_sharpen(st->d_frame_curr, W, H,
                       st->sharp_strength, st->stream);

    // Step 4: RAFT inference
    if (st->have_prev) {
        if (!st->raft.infer(st->d_frame_prev, st->d_frame_curr,
                            st->d_flow, st->stream)) {
            std::fprintf(stderr, "[raft_of] infer failed frame %u\n", st->frame_id);
        } else {
            const int rx0 = (int)(st->roi_x0 * W), rx1 = (int)(st->roi_x1 * W);
            const int ry0 = (int)(st->roi_y0 * H), ry1 = (int)(st->roi_y1 * H);

            // Step 5: GPU reduction on raw flow → mean_u, mean_v
            flow_reduce(st->d_flow, W, H,
                        rx0, rx1, ry0, ry1,
                        st->step, st->d_result, st->stream);

            FlowResult res{};
            cudaMemcpyAsync(&res, st->d_result, sizeof(FlowResult),
                            cudaMemcpyDeviceToHost, st->stream);
            cudaStreamSynchronize(st->stream);

            // Step 6: axis mapping for the TILTED mounting.
            // Forward motion now appears in the VERTICAL flow (mean_v);
            // lateral / yaw motion in the HORIZONTAL flow (mean_u).
            // (With the old mount, parallel to the ground, it was the opposite.)
            const float fwd_flow = res.mean_v;   // forward  -> vertical flow
            const float lat_flow = res.mean_u;   // lateral  -> horizontal flow

            // FOE correction now applies to the LATERAL axis (mean_u),
            // scaled by the forward signal (mean_v):
            //   lat_corrected = mean_u - (foe_a * mean_v + foe_b)
            const float lat_foe = lat_flow - (st->foe_a * fwd_flow + st->foe_b);

            // Step 7: convert px/frame → km/h
            const float vx_raw = -fwd_flow * st->scale_kmh;   // forward (+ = ahead)
            const float vy_raw =  lat_foe  * st->scale_kmh;   // lateral

            // Step 8: startup filter
            bool emit = true;

            if (!st->locked) {
                const bool valid_abs  = (vx_raw >= st->min_vx_kmh) &&
                                        (fabsf(vy_raw) <= st->max_vy_kmh);
                const int  gap        = (int)st->frame_id - (int)st->last_valid_frame;
                const bool ref_fresh  = (st->last_valid_vx >= 0.0f) &&
                                        (gap <= st->max_valid_gap);
                const float dvx       = ref_fresh ? fabsf(vx_raw - st->last_valid_vx) : 0.0f;
                const bool valid_rate = (dvx <= st->max_dvx_kmh);
                const bool valid      = valid_abs && valid_rate;

                if (!valid) {
                    st->consecutive_valid = 0;
                    emit = false;
                    std::fprintf(stderr,
                        "[raft_of] skip frame %u — vx=%.2f vy=%.2f dvx=%.2f km/h"
                        " (gap=%d ref_fresh=%d)\n",
                        st->frame_id, vx_raw, vy_raw, dvx, gap, (int)ref_fresh);
                } else {
                    st->last_valid_vx    = vx_raw;
                    st->last_valid_frame = st->frame_id;
                    st->consecutive_valid++;

                    if (st->consecutive_valid >= st->min_consecutive) {
                        st->locked = true;
                        std::fprintf(stderr,
                            "[raft_of] LOCKED at frame %u after %d consecutive valid frames\n",
                            st->frame_id, st->min_consecutive);
                    } else {
                        emit = false;
                        std::fprintf(stderr,
                            "[raft_of] hold frame %u — vx=%.2f vy=%.2f km/h"
                            " (consecutive=%d/%d)\n",
                            st->frame_id, vx_raw, vy_raw,
                            st->consecutive_valid, st->min_consecutive);
                    }
                }
            }

            if (emit) {
                const float vx_kmh = st->ema_vx.push(vx_raw);
                const float vy_kmh = st->ema_vy.push(vy_raw);

                std::fprintf(stdout,
                    "frame=%u  vx=%.3f  vy=%.3f  km/h\n",
                    st->frame_id, vx_kmh, vy_kmh);
                std::fflush(stdout);

                if (st->csv_file) {
                    // Raw flow components logged unmapped (mean_u, mean_v) so
                    // post-processing has the full signal; vx/vy are mapped+EMA.
                    std::fprintf(st->csv_file, "%u,%.4f,%.4f,%.4f,%.4f\n",
                        st->frame_id, res.mean_u, res.mean_v, vx_kmh, vy_kmh);
                    std::fflush(st->csv_file);
                }
            }

            // Step 9: FOE correction (visual) — in-place on d_flow.
            // NOTE: this kernel still corrects the v-channel of the flow field;
            // it is a no-op while FOE_A==FOE_B==0. When FOE is recalibrated on
            // the lateral (u) channel, update foe_correct_flow() accordingly.
            foe_correct_flow(st->d_flow, H, W,
                             st->foe_a, st->foe_b, st->stream);

            // Step 10: overlay — arrow field
            overlay_draw_flow(
                egl.d_y, egl.d_uv,
                egl.pitchY, egl.pitchUV,
                W, H, st->d_flow,
                st->roi_x0, st->roi_x1,
                st->roi_y0, st->roi_y1,
                st->step, st->arrow_scale, st->min_mag,
                st->stream);

            // Step 11: overlay — resultant vector (image-space flow:
            // horizontal = lateral, vertical = forward)
            overlay_draw_resultant(
                egl.d_y, egl.d_uv,
                egl.pitchY, egl.pitchUV,
                W, H,
                lat_foe, fwd_flow,
                st->roi_x0, st->roi_x1,
                st->roi_y0, st->roi_y1,
                st->result_scale, st->stream);
        }
    }

    cudaStreamSynchronize(st->stream);
    egl_unmap(egl);

    float *tmp       = st->d_frame_prev;
    st->d_frame_prev = st->d_frame_curr;
    st->d_frame_curr = tmp;

    st->have_prev = true;
    st->frame_id++;
}

static void pre_process(
    void**, unsigned int *inW, unsigned int *inH,
    unsigned int*, unsigned int*, ColorFormat*,
    unsigned int, void **userPtr)
{
    State *st = get_state(userPtr);
    if (inW && inH && *inW > 0 && *inH > 0) {
        st->W = (int)*inW;
        st->H = (int)*inH;
    }
}

static void post_process(
    void**, unsigned int*, unsigned int*,
    unsigned int*, unsigned int*, ColorFormat*,
    unsigned int, void **userPtr)
{
    if (!userPtr || !*userPtr) return;
    State *st = reinterpret_cast<State*>(*userPtr);
    if (st->csv_file)     { std::fclose(st->csv_file);     }
    if (st->d_frame_prev) { cudaFree(st->d_frame_prev);    }
    if (st->d_frame_curr) { cudaFree(st->d_frame_curr);    }
    if (st->d_flow)       { cudaFree(st->d_flow);          }
    if (st->d_result)     { cudaFree(st->d_result);        }
    if (st->stream)       { cudaStreamDestroy(st->stream); }
    std::fprintf(stderr, "[raft_of] Shutdown — %u frames\n", st->frame_id);
    delete st; *userPtr = nullptr;
}

extern "C" void init(CustomerFunction *f) {
    if (!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;
}
extern "C" void deinit(void) {}