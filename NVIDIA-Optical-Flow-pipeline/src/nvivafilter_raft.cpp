// nvivafilter_raft.cpp — RAFT zero-copy pipeline
// Output: mean displacement in x (u) and y (v) in px/frame

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

static int   env_int  (const char *k, int   d) { const char *v=getenv(k); return v?atoi(v):d; }
static float env_float(const char *k, float d) { const char *v=getenv(k); return v?atof(v):d; }
static const char *env_str(const char *k, const char *d) { const char *v=getenv(k); return v?v:d; }


// ── Exponential Moving Average ────────────────────────────────────────────────
// output[t] = α × input[t] + (1-α) × output[t-1]
// α = 1/(N+1) per equivalenza con SMA di finestra N
struct MovingAverage {
    float alpha  = 1.0f;   // α=1 → nessun filtro
    float value  = 0.0f;
    bool  inited = false;

    void init(int window) {
        // Converte finestra SMA in alpha EMA equivalente
        alpha = window <= 1 ? 1.0f : 2.0f / (window + 1.0f);
    }

    float update(float val) {
        if (!inited) { value = val; inited = true; return val; }
        value = alpha * val + (1.0f - alpha) * value;
        return value;
    }
};


// ── Rate Limiter ──────────────────────────────────────────────────────────────
// Rejects samples that change more than max_delta per frame
struct RateLimiter {
    float max_delta = 1e9f;  // km/h per frame
    float last      = 0.0f;
    bool  inited    = false;

    void init(float delta) { max_delta = delta; }

    float update(float val) {
        if (!inited) { last = val; inited = true; return val; }
        float delta = val - last;
        if (delta >  max_delta) val = last + max_delta;
        if (delta < -max_delta) val = last - max_delta;
        last = val;
        return val;
    }
};

struct State {
    bool  inited      = false;
    bool  init_failed = false;
    int   W = 0, H = 0;

    // ROI in coordinate normalizzate [0,1]
    float roi_x0 = 0.55f;
    float roi_x1 = 0.95f;
    float roi_y0 = 0.45f;
    float roi_y1 = 0.68f;

    // Overlay
    int   step         = 16;   // distanza tra frecce in pixel
    float arrow_scale  = 4.0f;
    float min_mag      = 1.5f;
    float result_scale = 8.0f;
    float sharp_strength = 1.5f;

    RaftInfer    raft;
    float       *d_frame_prev = nullptr;
    float       *d_frame_curr = nullptr;
    float       *d_flow       = nullptr;
    FlowResult  *d_result     = nullptr;

    bool         have_prev    = false;
    cudaStream_t stream       = nullptr;
    uint32_t     frame_id     = 0;
    uint32_t     warmup_frames = 0;  // skip first N frames (overexposed)

    FILE        *csv_file  = nullptr;

    MovingAverage ma_u;
    MovingAverage ma_v;

    RateLimiter   rl_u;
    RateLimiter   rl_v;

    float foe_a = 0.0f;  // v = a*u + b
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

    st->roi_x0       = env_float("RAFT_ROI_X0",      st->roi_x0);
    st->roi_x1       = env_float("RAFT_ROI_X1",      st->roi_x1);
    st->roi_y0       = env_float("RAFT_ROI_Y0",      st->roi_y0);
    st->roi_y1       = env_float("RAFT_ROI_Y1",      st->roi_y1);
    st->step         = env_int  ("RAFT_STEP",         st->step);
    st->arrow_scale  = env_float("RAFT_ARROW_SCALE",  st->arrow_scale);
    st->min_mag      = env_float("RAFT_MIN_MAG",      st->min_mag);
    st->result_scale = env_float("RAFT_RESULT_SCALE", st->result_scale);
    st->sharp_strength  = env_float("RAFT_SHARP",      st->sharp_strength);
    st->warmup_frames   = (uint32_t)env_int("RAFT_WARMUP_FRAMES", 20);

    const int ma_win   = env_int("RAFT_MA_WINDOW",   1);
    const int ma_win_v = env_int("RAFT_MA_WINDOW_V", ma_win);
    st->ma_u.init(ma_win);
    st->ma_v.init(ma_win_v);

    st->foe_a = env_float("RAFT_FOE_A", 0.0f);
    st->foe_b = env_float("RAFT_FOE_B", 0.0f);

    // Rate limiter: max change per frame in px/frame
    // Default 2 km/h → px/frame: 2 / (1/px_per_m * 100 * 3.6)
    // At PX_PER_M=428: 2 / 0.849 = 2.356 px/frame
    const float rl_px = env_float("RAFT_RATE_LIMIT_KMPH", 2.0f) / ((1.0f/428.0f) * 100.0f * 3.6f);
    st->rl_u.init(rl_px);
    st->rl_v.init(rl_px);

    const char *engine_path = env_str("RAFT_ENGINE_PATH",
        "/home/ntc-orin/raft/raft_large_fp16.engine");

    // Calcola N = numero di punti nella ROI
    const int rx0 = (int)(st->roi_x0 * W), rx1 = (int)(st->roi_x1 * W);
    const int ry0 = (int)(st->roi_y0 * H), ry1 = (int)(st->roi_y1 * H);
    int nx = 0, ny = 0;
    for (int x = rx0; x < rx1; x += st->step) nx++;
    for (int y = ry0; y < ry1; y += st->step) ny++;
    const int N = nx * ny;

    std::fprintf(stderr,
        "[raft_of] Init W=%d H=%d engine=%s\n"
        "[raft_of] ROI px=[%d,%d]x[%d,%d]  step=%d  N=%d punti\n"
        "[raft_of] mean_u = sum(u_i) / %d,  mean_v = sum(v_i) / %d\n",
        W, H, engine_path,
        rx0, rx1, ry0, ry1, st->step, N, N, N);

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

    // Apri CSV
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

    // Step 1: mappa EGLImage → CUDA pointer (zero-copy)
    EGLMapResult egl;
    if (!egl_map(image, W, H, egl)) {
        std::fprintf(stderr, "[raft_of] egl_map failed frame %u\n", st->frame_id);
        return;
    }

    // Step 2: NV12 → float32 RGB [1,3,H,W]
    nv12_to_rgb_fp16(egl.d_y, egl.d_uv,
                     egl.pitchY, egl.pitchUV,
                     W, H, st->d_frame_curr, st->stream);

    // Step 3: sharpening (opzionale, RAFT_SHARP=0 per disabilitare)
    preprocess_sharpen(st->d_frame_curr, W, H,
                       st->sharp_strength, st->stream);

    // Step 4: inferenza RAFT
    if (st->have_prev) {
        if (!st->raft.infer(st->d_frame_prev, st->d_frame_curr,
                            st->d_flow, st->stream)) {
            std::fprintf(stderr, "[raft_of] infer failed frame %u\n", st->frame_id);
        } else {
            const int rx0=(int)(st->roi_x0*W), rx1=(int)(st->roi_x1*W);
            const int ry0=(int)(st->roi_y0*H), ry1=(int)(st->roi_y1*H);

            // Step 5: riduzione GPU → mean_u e mean_v sulla ROI
            // Kernel: accumula sum_u += flow[y,x,0] e sum_v += flow[y,x,1]
            //         per ogni punto (x,y) con passo step nella ROI
            //         poi divide per N = numero di punti
            flow_reduce(st->d_flow, W, H,
                        rx0, rx1, ry0, ry1,
                        st->step, st->d_result, st->stream);

            // Copia ~8 byte da GPU a CPU
            FlowResult res{};
            cudaMemcpyAsync(&res, st->d_result, sizeof(FlowResult),
                            cudaMemcpyDeviceToHost, st->stream);
            cudaStreamSynchronize(st->stream);

            // Stampa displacement medio in px/frame
            // Correzione FOE: sottrai componente geometrica da mean_v
            const float v_foe_corrected = res.mean_v - (st->foe_a * res.mean_u + st->foe_b);

            // Rate limiter: rigetta salti impossibili (default 2 km/h per frame)
            const float u_rl = st->rl_u.update(res.mean_u);
            const float v_rl = st->rl_v.update(v_foe_corrected);

            const float u_filt = st->ma_u.update(u_rl);
            const float v_filt = st->ma_v.update(v_rl);  // MA leggera su v

            // Conversione px/frame → km/h
            // v_kmh = (px/frame) / (px/m) * FPS * 3.6
            const float px_per_m = env_float("RAFT_PX_PER_M", 424.0f);
            const float SCALE = (1.0f / px_per_m) * 100.0f * 3.6f;
            // During warmup (overexposed frames) output zero
            const bool  in_warmup = (st->frame_id < st->warmup_frames);
            const float vx_kmh = in_warmup ? 0.0f : -u_filt * SCALE;
            const float vy_min_vx = env_float("RAFT_VY_MIN_VX", 5.0f);
            const float vy_kmh = (in_warmup || vx_kmh < vy_min_vx) ? 0.0f : v_filt * SCALE;

            std::fprintf(stdout,
                "frame=%u  vx=%.3f  vy=%.3f  km/h\n",
                st->frame_id, vx_kmh, vy_kmh);
            std::fflush(stdout);

            if (st->csv_file) {
                std::fprintf(st->csv_file, "%u,%.4f,%.4f,%.4f,%.4f\n",
                    st->frame_id, u_filt, v_filt, vx_kmh, vy_kmh);
                std::fflush(st->csv_file);
            }

            // Overlay: frecce verdi + vettore risultante blu
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

    // Swap buffer prev/curr
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
        st->W=(int)*inW; st->H=(int)*inH;
    }
}

static void post_process(
    void**, unsigned int*, unsigned int*,
    unsigned int*, unsigned int*, ColorFormat*,
    unsigned int, void **userPtr)
{
    if (!userPtr || !*userPtr) return;
    State *st = reinterpret_cast<State*>(*userPtr);
    if (st->csv_file)     { std::fclose(st->csv_file);  }
    if (st->d_frame_prev) { cudaFree(st->d_frame_prev); }
    if (st->d_frame_curr) { cudaFree(st->d_frame_curr); }
    if (st->d_flow)       { cudaFree(st->d_flow);       }
    if (st->d_result)     { cudaFree(st->d_result);     }
    if (st->stream)       { cudaStreamDestroy(st->stream); }
    std::fprintf(stderr, "[raft_of] Shutdown — %u frames\n", st->frame_id);
    delete st; *userPtr=nullptr;
}

extern "C" void init(CustomerFunction *f) {
    if (!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;
}
extern "C" void deinit(void) {}