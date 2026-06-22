// nvivafilter_raft.cpp — RAFT zero-copy optical flow pipeline
// Estimates longitudinal (vx) and lateral (vy) vehicle speed in km/h
// from raw camera frames decoded by NVDEC on Jetson Orin AGX.
//
// Processing chain:
//   EGLImage (NVMM) → NV12→float32 → Sharpen → RAFT TRT → flow_reduce
//   → axis mapping (TILTED mount) → Startup filter → EMA
//   → beta confidence from forward pixel displacement → CSV + stdout → overlay
//
// ── AXIS MAPPING (tilted mount, 2026-06) ───────────────────────────────────
//   The camera is no longer parallel to the ground. With this mounting the
//   vehicle's FORWARD motion appears as VERTICAL image flow (mean_v), and the
//   lateral / yaw motion appears as HORIZONTAL image flow (mean_u).
//
//     fwd_px = -mean_v
//     lat_px = -mean_u
//
//     vx (forward) = fwd_px * scale_kmh
//     vy (lateral) = lat_px * scale_kmh
//
//   NOTE: vy derives from mean_u, which may also pick up yaw rotation at low
//   speed. Beta is therefore confidence-weighted when forward displacement is
//   small, instead of inventing a lateral value.
//
// ── BETA LOGIC ─────────────────────────────────────────────────────────────
//   beta_raw_deg is calculated directly from optical-flow pixel displacement:
//
//     beta_raw_deg = atan2(lat_px, fwd_px) * 180/pi
//
//   When |fwd_px| is small, beta is not observable/reliable. Instead of using
//   a hard rule that forces lat_px to an artificial value, we compute a smooth
//   confidence from |fwd_px|:
//
//     |fwd_px| <= RAFT_BETA_FWD_LOW_PX   → confidence = 0
//     |fwd_px| >= RAFT_BETA_FWD_HIGH_PX  → confidence = 1
//     between them                      → smooth transition
//
//     beta_candidate_deg = beta_confidence * beta_raw_deg
//
//   Then a beta-only spike limiter clamps unrealistic frame-to-frame jumps:
//
//     beta_px_deg = previous_beta + clamp(candidate - previous_beta,
//                                        -RAFT_BETA_MAX_STEP_DEG,
//                                        +RAFT_BETA_MAX_STEP_DEG)
//
//   The confidence and spike limiter are applied ONLY to beta. vx/vy and raw
//   flow remain untouched.
//
// ── SPEED SCALE ────────────────────────────────────────────────────────────
//   scale_kmh = (1 / PX_PER_M) * FPS * 3.6
//   FPS is configurable via RAFT_FPS.
//
// ── STARTUP FILTER ─────────────────────────────────────────────────────────
//   Active until first stable lock:
//     Gate 1 — absolute bounds (vx >= MIN_VX, |vy| <= MAX_VY)
//     Gate 2 — gap-aware derivative bound (|dvx| <= MAX_DVX)
//     Gate 3 — consecutive valid frames (MIN_CONSECUTIVE in a row)
//   Once locked, all frames are emitted unconditionally.
//
// ── EMA FILTER ─────────────────────────────────────────────────────────────
//   vx: alpha = RAFT_ALPHA_VX (default 0.4)
//   vy: alpha = RAFT_ALPHA_VY (default 0.2)
//   Set RAFT_ALPHA_VX=1.0 / RAFT_ALPHA_VY=1.0 to disable.

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
#include <cstdint>

static int         env_int  (const char *k, int   d) { const char *v = getenv(k); return v ? atoi(v) : d; }
static float       env_float(const char *k, float d) { const char *v = getenv(k); return v ? atof(v) : d; }
static const char *env_str  (const char *k, const char *d) { const char *v = getenv(k); return v ? v : d; }

static float clamp01(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

static float smoothstep01(float x) {
    x = clamp01(x);
    return x * x * (3.0f - 2.0f * x);
}

// ── EMA filter ────────────────────────────────────────────────────────────────
struct EMA {
    float alpha  = 1.0f;
    float value  = 0.0f;
    bool  inited = false;

    void init(float a) { alpha = a; }

    float push(float x) {
        if (!inited) {
            value = x;
            inited = true;
            return x;
        }
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
    // scale_kmh = (1 / px_per_m) * fps * 3.6
    float scale_kmh = 0.0f;

    // Beta confidence rule.
    // Beta is progressively suppressed when forward pixel displacement is too small.
    // Applied ONLY to beta, never to vx/vy or raw flow.
    float beta_fwd_low_px  = 2.0f;
    float beta_fwd_high_px = 8.0f;

    // Beta spike limiter.
    // Applied ONLY to final beta, never to vx/vy or raw flow.
    float beta_max_step_deg   = 2.0f;
    float beta_spike_conf_min = 0.2f;
    float beta_prev_deg       = 0.0f;
    bool  beta_prev_valid     = false;

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
    // separator regardless of the system locale.
    std::setlocale(LC_NUMERIC, "C");

    st->W = W;
    st->H = H;

    st->roi_x0          = env_float("RAFT_ROI_X0",          st->roi_x0);
    st->roi_x1          = env_float("RAFT_ROI_X1",          st->roi_x1);
    st->roi_y0          = env_float("RAFT_ROI_Y0",          st->roi_y0);
    st->roi_y1          = env_float("RAFT_ROI_Y1",          st->roi_y1);
    st->step            = env_int  ("RAFT_STEP",            st->step);
    st->arrow_scale     = env_float("RAFT_ARROW_SCALE",     st->arrow_scale);
    st->min_mag         = env_float("RAFT_MIN_MAG",         st->min_mag);
    st->result_scale    = env_float("RAFT_RESULT_SCALE",    st->result_scale);
    st->sharp_strength  = env_float("RAFT_SHARP",           st->sharp_strength);

    st->min_vx_kmh      = env_float("RAFT_MIN_VX_KMH",      st->min_vx_kmh);
    st->max_vy_kmh      = env_float("RAFT_MAX_VY_KMH",      st->max_vy_kmh);
    st->max_dvx_kmh     = env_float("RAFT_MAX_DVX_KMH",     st->max_dvx_kmh);
    st->max_valid_gap   = env_int  ("RAFT_MAX_VALID_GAP",   st->max_valid_gap);
    st->min_consecutive = env_int  ("RAFT_MIN_CONSECUTIVE", st->min_consecutive);

    st->ema_vx.init(env_float("RAFT_ALPHA_VX", 0.4f));
    st->ema_vy.init(env_float("RAFT_ALPHA_VY", 0.2f));

    st->beta_fwd_low_px  = env_float("RAFT_BETA_FWD_LOW_PX",  st->beta_fwd_low_px);
    st->beta_fwd_high_px = env_float("RAFT_BETA_FWD_HIGH_PX", st->beta_fwd_high_px);

    st->beta_max_step_deg =
        env_float("RAFT_BETA_MAX_STEP_DEG", st->beta_max_step_deg);
    st->beta_spike_conf_min =
        env_float("RAFT_BETA_SPIKE_CONF_MIN", st->beta_spike_conf_min);

    if (st->beta_max_step_deg < 0.0f) {
        std::fprintf(stderr,
            "[raft_of] WARNING: RAFT_BETA_MAX_STEP_DEG < 0. Forcing to 0.0\n");
        st->beta_max_step_deg = 0.0f;
    }

    st->beta_spike_conf_min = clamp01(st->beta_spike_conf_min);

    if (st->beta_fwd_high_px <= st->beta_fwd_low_px) {
        std::fprintf(stderr,
            "[raft_of] WARNING: RAFT_BETA_FWD_HIGH_PX <= RAFT_BETA_FWD_LOW_PX. "
            "Forcing high = low + 1.0\n");
        st->beta_fwd_high_px = st->beta_fwd_low_px + 1.0f;
    }

    // Speed scale — read once at init, not per frame.
    const float px_per_m = env_float("RAFT_PX_PER_M", 55.0f);
    const float fps      = env_float("RAFT_FPS",      30.0f);
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
        "[raft_of] AXIS MAP: fwd_px=-mean_v, lat_px=-mean_u (TILTED mount)\n"
        "[raft_of] ROI px=[%d,%d]x[%d,%d]  step=%d  N=%d points\n"
        "[raft_of] startup gate 1 (absolute)   : vx >= %.1f km/h, |vy| <= %.1f km/h\n"
        "[raft_of] startup gate 2 (derivative) : |dvx| <= %.1f km/h/frame, gap <= %d frames\n"
        "[raft_of] startup gate 3 (consecutive): %d valid frames in a row to lock\n"
        "[raft_of] EMA filter                  : alpha_vx=%.2f  alpha_vy=%.2f\n"
        "[raft_of] Beta confidence             : low=%.3f px  high=%.3f px  smoothstep(|fwd_px|)\n"
        "[raft_of] Beta spike limiter          : max_step=%.3f deg/frame  conf_min=%.3f\n"
        "[raft_of] Speed scale                 : px_per_m=%.1f fps=%.1f → %.6f (km/h per px/frame)\n",
        W, H, engine_path,
        rx0, rx1, ry0, ry1, st->step, nx * ny,
        st->min_vx_kmh, st->max_vy_kmh,
        st->max_dvx_kmh, st->max_valid_gap,
        st->min_consecutive,
        st->ema_vx.alpha, st->ema_vy.alpha,
        st->beta_fwd_low_px, st->beta_fwd_high_px,
        st->beta_max_step_deg, st->beta_spike_conf_min,
        px_per_m, fps, st->scale_kmh);

    if (cudaStreamCreateWithFlags(&st->stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "[raft_of] cudaStreamCreate failed\n");
        st->init_failed = true;
        return false;
    }

    if (!st->raft.init(engine_path, W, H)) {
        std::fprintf(stderr, "[raft_of] RAFT init failed\n");
        st->init_failed = true;
        return false;
    }

    const size_t frame_bytes = sizeof(float) * 3 * H * W;
    const size_t flow_bytes  = sizeof(float) * 2 * H * W;

    if (cudaMalloc(&st->d_frame_prev, frame_bytes) != cudaSuccess ||
        cudaMalloc(&st->d_frame_curr, frame_bytes) != cudaSuccess ||
        cudaMalloc(&st->d_flow,       flow_bytes)  != cudaSuccess ||
        cudaMalloc(&st->d_result, sizeof(FlowResult)) != cudaSuccess) {
        std::fprintf(stderr, "[raft_of] cudaMalloc failed\n");
        st->init_failed = true;
        return false;
    }

    cudaMemset(st->d_frame_prev, 0, frame_bytes);
    cudaMemset(st->d_frame_curr, 0, frame_bytes);
    cudaMemset(st->d_flow,       0, flow_bytes);
    cudaMemset(st->d_result,     0, sizeof(FlowResult));

    const char *csv_path = env_str("RAFT_CSV_PATH", "/home/jetson-ntc/raft/output.csv");
    st->csv_file = std::fopen(csv_path, "w");
    if (!st->csv_file) {
        std::fprintf(stderr, "[raft_of] Cannot open CSV: %s\n", csv_path);
        st->init_failed = true;
        return false;
    }

    std::fprintf(st->csv_file,
        "frame,mean_u_px,mean_v_px,fwd_px,lat_px,vx_kmh,vy_kmh,beta_raw_deg,beta_conf,beta_candidate_deg,beta_px_deg,beta_spike_limited\n");
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

    // Step 4: RAFT inference between previous and current frame
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
            // Forward motion appears in vertical flow mean_v.
            // Lateral/yaw motion appears in horizontal flow mean_u.
            const float fwd_flow = res.mean_v;
            const float lat_flow = res.mean_u;

            // Signed pixel displacement in vehicle convention.
            //   fwd_px > 0  => vehicle moving forward
            //   lat_px > 0  => vehicle moving right
            const float fwd_px = -fwd_flow;
            const float lat_px = -lat_flow;

            // Step 7: convert px/frame → km/h.
            // IMPORTANT: vx/vy are not affected by beta confidence.
            const float vx_raw = fwd_px * st->scale_kmh;
            const float vy_raw = lat_px * st->scale_kmh;

            // Step 8: beta from pixel displacement, with smooth confidence.
            // beta_raw_deg is the direct optical-flow beta.
            // beta_conf suppresses beta when forward displacement is too small.
            const float beta_raw_deg = atan2f(lat_px, fwd_px) * 57.295779513f;
            const float abs_fwd_px = fabsf(fwd_px);

            const float beta_conf_linear =
                (abs_fwd_px - st->beta_fwd_low_px) /
                (st->beta_fwd_high_px - st->beta_fwd_low_px);

            const float beta_conf = smoothstep01(beta_conf_linear);
            const float beta_candidate_deg = beta_conf * beta_raw_deg;

            // Step 8b: beta-only spike limiter.
            // This limits unrealistic frame-to-frame jumps in final beta.
            // It does NOT modify mean_u, mean_v, vx, vy, or beta_raw_deg.
            float beta_px_deg = beta_candidate_deg;
            bool beta_spike_limited = false;

            if (beta_conf >= st->beta_spike_conf_min) {
                if (st->beta_prev_valid) {
                    const float delta = beta_candidate_deg - st->beta_prev_deg;

                    if (fabsf(delta) > st->beta_max_step_deg) {
                        const float limited_delta =
                            copysignf(st->beta_max_step_deg, delta);
                        beta_px_deg = st->beta_prev_deg + limited_delta;
                        beta_spike_limited = true;
                    }
                }

                st->beta_prev_deg = beta_px_deg;
                st->beta_prev_valid = true;
            } else {
                // Low confidence: beta is already suppressed by beta_conf.
                // Do not apply spike limiting here. Keep the previous valid
                // reference only while there is still some confidence, and
                // reset it when confidence is essentially zero.
                beta_px_deg = beta_candidate_deg;

                if (beta_conf <= 0.001f) {
                    st->beta_prev_deg = 0.0f;
                    st->beta_prev_valid = false;
                }
            }

            // Step 9: startup filter for vx/vy emission
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
                    "frame=%u  vx=%.3f km/h  vy=%.3f km/h  beta=%.3f deg  beta_raw=%.3f deg  conf=%.3f  limited=%d\n",
                    st->frame_id, vx_kmh, vy_kmh, beta_px_deg, beta_raw_deg, beta_conf,
                    beta_spike_limited ? 1 : 0);
                std::fflush(stdout);

                if (st->csv_file) {
                    // mean_u/mean_v are raw unmapped optical-flow components.
                    // fwd_px/lat_px are signed vehicle-convention pixel signals.
                    // vx/vy are EMA-filtered km/h values.
                    // beta_raw_deg is the direct pixel angle.
                    // beta_candidate_deg is beta after confidence weighting.
                    // beta_px_deg is final beta after optional spike limiting.
                    std::fprintf(st->csv_file,
                        "%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                        st->frame_id,
                        res.mean_u,
                        res.mean_v,
                        fwd_px,
                        lat_px,
                        vx_kmh,
                        vy_kmh,
                        beta_raw_deg,
                        beta_conf,
                        beta_candidate_deg,
                        beta_px_deg,
                        beta_spike_limited ? 1 : 0);
                    std::fflush(st->csv_file);
                }
            }

            // Step 10: overlay — arrow field, using raw flow field
            overlay_draw_flow(
                egl.d_y, egl.d_uv,
                egl.pitchY, egl.pitchUV,
                W, H, st->d_flow,
                st->roi_x0, st->roi_x1,
                st->roi_y0, st->roi_y1,
                st->step, st->arrow_scale, st->min_mag,
                st->stream);

            // Step 11: overlay — resultant vector in image-space flow
            // horizontal = lateral, vertical = forward
            overlay_draw_resultant(
                egl.d_y, egl.d_uv,
                egl.pitchY, egl.pitchUV,
                W, H,
                lat_flow, fwd_flow,
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
    delete st;
    *userPtr = nullptr;
}

extern "C" void init(CustomerFunction *f) {
    if (!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;
}

extern "C" void deinit(void) {}