// nvivafilter_raft.cpp — RAFT zero-copy optical flow pipeline
// Estimates longitudinal (vx) and lateral (vy) vehicle speed in km/h
// from raw camera frames decoded by NVDEC on Jetson Orin AGX.
//
// Processing chain:
//   EGLImage (NVMM) → NV12→float32 → Sharpen → RAFT TRT → flow_reduce
//   → FOE correction → startup filter → CSV + stdout
//
// Startup filter (active until first stable lock):
//
//   Gate 1 — absolute bounds
//     Rejects frames where vx < RAFT_MIN_VX_KMH or |vy| > RAFT_MAX_VY_KMH.
//     Catches AEC transients and corrupted RAFT inputs.
//
//   Gate 2 — gap-aware derivative bound
//     Rejects frames where |vx - last_valid_vx| > RAFT_MAX_DVX_KMH.
//     Disabled when the last valid reference is older than RAFT_MAX_VALID_GAP
//     frames, so a burst of rejections cannot permanently lock out valid data.
//
//   Gate 3 — consecutive valid frames
//     Suppresses output until RAFT_MIN_CONSECUTIVE valid frames are seen in a
//     row. Prevents isolated transients that pass gates 1+2 from being emitted.
//     The derivative reference is updated throughout to stay fresh.
//
// Once RAFT_MIN_CONSECUTIVE consecutive valid frames have been seen, the
// pipeline is considered "locked". From that point all three gates are
// permanently disabled and every frame is emitted unconditionally — including
// during maneuvers where vx may briefly dip below the startup threshold.

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

static int         env_int  (const char *k, int   d) { const char *v=getenv(k); return v?atoi(v):d; }
static float       env_float(const char *k, float d) { const char *v=getenv(k); return v?atof(v):d; }
static const char *env_str  (const char *k, const char *d) { const char *v=getenv(k); return v?v:d; }


// ── Pipeline State ────────────────────────────────────────────────────────────
struct State {
    bool  inited      = false;
    bool  init_failed = false;
    int   W = 0, H = 0;

    // ROI in normalized coordinates [0,1]
    float roi_x0 = 0.55f;
    float roi_x1 = 0.95f;
    float roi_y0 = 0.45f;
    float roi_y1 = 0.68f;

    // Overlay parameters
    int   step           = 16;
    float arrow_scale    = 4.0f;
    float min_mag        = 1.5f;
    float result_scale   = 8.0f;
    float sharp_strength = 1.5f;

    // Startup gate 1 — absolute bounds
    float min_vx_kmh = 5.0f;   // RAFT_MIN_VX_KMH
    float max_vy_kmh = 3.0f;   // RAFT_MAX_VY_KMH

    // Startup gate 2 — gap-aware derivative bound
    float    max_dvx_kmh      = 5.0f;   // RAFT_MAX_DVX_KMH
    int      max_valid_gap    = 3;      // RAFT_MAX_VALID_GAP
    float    last_valid_vx    = -1.0f;  // -1 = no valid sample yet
    uint32_t last_valid_frame = 0;

    // Startup gate 3 — consecutive valid frames before first output
    int  min_consecutive   = 3;   // RAFT_MIN_CONSECUTIVE
    int  consecutive_valid = 0;

    // Lock flag — set permanently after first stable lock.
    // All startup gates are bypassed once locked.
    bool locked = false;

    RaftInfer    raft;
    float       *d_frame_prev = nullptr;
    float       *d_frame_curr = nullptr;
    float       *d_flow       = nullptr;
    FlowResult  *d_result     = nullptr;

    bool         have_prev = false;
    cudaStream_t stream    = nullptr;
    uint32_t     frame_id  = 0;

    FILE        *csv_file = nullptr;

    // Focus of Expansion correction coefficients
    // mean_v_corrected = mean_v - (foe_a * mean_u + foe_b)
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

    st->foe_a = env_float("RAFT_FOE_A", 0.0f);
    st->foe_b = env_float("RAFT_FOE_B", 0.0f);

    const char *engine_path = env_str("RAFT_ENGINE_PATH",
        "/home/ntc-orin/raft/raft_large_fp16.engine");

    const int rx0 = (int)(st->roi_x0 * W), rx1 = (int)(st->roi_x1 * W);
    const int ry0 = (int)(st->roi_y0 * H), ry1 = (int)(st->roi_y1 * H);
    int nx = 0, ny = 0;
    for (int x = rx0; x < rx1; x += st->step) nx++;
    for (int y = ry0; y < ry1; y += st->step) ny++;
    const int N = nx * ny;

    std::fprintf(stderr,
        "[raft_of] Init W=%d H=%d engine=%s\n"
        "[raft_of] ROI px=[%d,%d]x[%d,%d]  step=%d  N=%d points\n"
        "[raft_of] mean_u = sum(u_i) / %d,  mean_v = sum(v_i) / %d\n"
        "[raft_of] startup gate 1 (absolute)   : vx >= %.1f km/h, |vy| <= %.1f km/h\n"
        "[raft_of] startup gate 2 (derivative) : |dvx| <= %.1f km/h/frame, gap <= %d frames\n"
        "[raft_of] startup gate 3 (consecutive): %d valid frames in a row to lock\n",
        W, H, engine_path,
        rx0, rx1, ry0, ry1, st->step, N, N, N,
        st->min_vx_kmh, st->max_vy_kmh,
        st->max_dvx_kmh, st->max_valid_gap,
        st->min_consecutive);

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

    const char *csv_path = env_str("RAFT_CSV_PATH", "/home/ntc-orin/raft/output.csv");
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

    // Step 3: unsharp mask sharpening (set RAFT_SHARP=0 to disable)
    preprocess_sharpen(st->d_frame_curr, W, H,
                       st->sharp_strength, st->stream);

    // Step 4: RAFT inference — produces flow field [1,2,H,W]
    if (st->have_prev) {
        if (!st->raft.infer(st->d_frame_prev, st->d_frame_curr,
                            st->d_flow, st->stream)) {
            std::fprintf(stderr, "[raft_of] infer failed frame %u\n", st->frame_id);
        } else {
            const int rx0 = (int)(st->roi_x0 * W), rx1 = (int)(st->roi_x1 * W);
            const int ry0 = (int)(st->roi_y0 * H), ry1 = (int)(st->roi_y1 * H);

            // Step 5: GPU reduction → mean_u, mean_v over ROI
            flow_reduce(st->d_flow, W, H,
                        rx0, rx1, ry0, ry1,
                        st->step, st->d_result, st->stream);

            FlowResult res{};
            cudaMemcpyAsync(&res, st->d_result, sizeof(FlowResult),
                            cudaMemcpyDeviceToHost, st->stream);
            cudaStreamSynchronize(st->stream);

            // Step 6: FOE correction
            // Removes spurious vertical flow caused by camera pitch.
            // mean_v_corrected = mean_v - (foe_a * mean_u + foe_b)
            const float v_foe = res.mean_v - (st->foe_a * res.mean_u + st->foe_b);

            // Step 7: convert px/frame → km/h
            // vx = -mean_u * (1/px_per_m) * FPS * 3.6  (forward motion → negative u)
            // vy =  v_foe  * (1/px_per_m) * FPS * 3.6
            const float px_per_m = env_float("RAFT_PX_PER_M", 424.0f);
            const float SCALE    = (1.0f / px_per_m) * 100.0f * 3.6f;
            const float vx_kmh   = -res.mean_u * SCALE;
            const float vy_kmh   =  v_foe      * SCALE;

            // Step 8: startup filter — only active before lock
            // Once locked, all frames are emitted unconditionally.
            bool emit = true;

            if (!st->locked) {
                // Gate 1: absolute bounds
                const bool valid_abs = (vx_kmh        >= st->min_vx_kmh) &&
                                       (fabsf(vy_kmh)  <= st->max_vy_kmh);

                // Gate 2: gap-aware derivative bound
                const int  gap       = (int)st->frame_id - (int)st->last_valid_frame;
                const bool ref_fresh = (st->last_valid_vx >= 0.0f) &&
                                       (gap <= st->max_valid_gap);
                const float dvx      = ref_fresh
                                       ? fabsf(vx_kmh - st->last_valid_vx)
                                       : 0.0f;
                const bool valid_rate = (dvx <= st->max_dvx_kmh);

                const bool valid = valid_abs && valid_rate;

                if (!valid) {
                    st->consecutive_valid = 0;
                    emit = false;
                    std::fprintf(stderr,
                        "[raft_of] skip frame %u — vx=%.2f vy=%.2f dvx=%.2f km/h"
                        " (gap=%d ref_fresh=%d)\n",
                        st->frame_id, vx_kmh, vy_kmh, dvx, gap, (int)ref_fresh);
                } else {
                    // Update derivative reference throughout startup
                    st->last_valid_vx    = vx_kmh;
                    st->last_valid_frame = st->frame_id;
                    st->consecutive_valid++;

                    if (st->consecutive_valid >= st->min_consecutive) {
                        // Stable signal confirmed — lock permanently
                        st->locked = true;
                        std::fprintf(stderr,
                            "[raft_of] LOCKED at frame %u after %d consecutive valid frames\n",
                            st->frame_id, st->min_consecutive);
                    } else {
                        // Valid but still in warmup — hold output
                        emit = false;
                        std::fprintf(stderr,
                            "[raft_of] hold frame %u — vx=%.2f vy=%.2f km/h"
                            " (consecutive=%d/%d)\n",
                            st->frame_id, vx_kmh, vy_kmh,
                            st->consecutive_valid, st->min_consecutive);
                    }
                }
            }

            if (emit) {
                std::fprintf(stdout,
                    "frame=%u  vx=%.3f  vy=%.3f  km/h\n",
                    st->frame_id, vx_kmh, vy_kmh);
                std::fflush(stdout);

                if (st->csv_file) {
                    std::fprintf(st->csv_file, "%u,%.4f,%.4f,%.4f,%.4f\n",
                        st->frame_id, res.mean_u, v_foe, vx_kmh, vy_kmh);
                    std::fflush(st->csv_file);
                }
            }

            // Overlay is always drawn — useful for visual debug on skipped frames too
            overlay_draw_flow(
                egl.d_y, egl.d_uv,
                egl.pitchY, egl.pitchUV,
                W, H, st->d_flow,
                st->roi_x0, st->roi_x1,
                st->roi_y0, st->roi_y1,
                st->step, st->arrow_scale, st->min_mag,
                st->stream);

            overlay_draw_resultant(
                egl.d_y, egl.d_uv,
                egl.pitchY, egl.pitchUV,
                W, H,
                res.mean_u, res.mean_v,
                st->roi_x0, st->roi_x1,
                st->roi_y0, st->roi_y1,
                st->result_scale, st->stream);
        }
    }

    cudaStreamSynchronize(st->stream);
    egl_unmap(egl);

    // Swap prev/curr frame buffers
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