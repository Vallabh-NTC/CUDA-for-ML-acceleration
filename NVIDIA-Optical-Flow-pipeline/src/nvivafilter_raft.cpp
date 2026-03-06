// nvivafilter_raft.cpp — RAFT zero-copy pipeline
// float32 I/O + sharpening + flow reduce + resultant overlay + CSV export

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

static int   env_int  (const char *k, int   d) { const char *v=getenv(k); return v?atoi(v):d; }
static float env_float(const char *k, float d) { const char *v=getenv(k); return v?atof(v):d; }
static const char *env_str(const char *k, const char *d) { const char *v=getenv(k); return v?v:d; }

// ── Constants ─────────────────────────────────────────────────────────────────
// Configurable via RAFT_PX_PER_M env var (default 454)
static float get_px_per_m() {
    const char *v = getenv("RAFT_PX_PER_M");
    return v ? atof(v) : 454.0f;
}
static constexpr float FPS = 100.0f;
// px/frame → m/s → km/h
// speed_kmh = (px/frame) / PX_PER_M * FPS * 3.6
// Computed at runtime from env

struct State {
    bool  inited      = false;
    bool  init_failed = false;
    int   W = 0, H = 0;

    float roi_x0       = 0.55f;
    float roi_x1       = 0.95f;
    float roi_y0       = 0.45f;
    float roi_y1       = 0.68f;

    int   step         = 16;
    float arrow_scale  = 4.0f;
    float min_mag      = 1.5f;
    float result_scale = 8.0f;
    float sharp_strength = 1.5f;

    RaftInfer raft;

    float      *d_frame_prev = nullptr;
    float      *d_frame_curr = nullptr;
    float      *d_flow       = nullptr;
    FlowResult *d_result     = nullptr;

    bool  have_prev  = false;
    cudaStream_t stream = nullptr;
    uint32_t frame_id   = 0;

    // CSV
    FILE *csv_file = nullptr;
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

    st->roi_x0        = env_float("RAFT_ROI_X0",      st->roi_x0);
    st->roi_x1        = env_float("RAFT_ROI_X1",      st->roi_x1);
    st->roi_y0        = env_float("RAFT_ROI_Y0",      st->roi_y0);
    st->roi_y1        = env_float("RAFT_ROI_Y1",      st->roi_y1);
    st->step          = env_int  ("RAFT_STEP",         st->step);
    st->arrow_scale   = env_float("RAFT_ARROW_SCALE",  st->arrow_scale);
    st->min_mag       = env_float("RAFT_MIN_MAG",      st->min_mag);
    st->result_scale  = env_float("RAFT_RESULT_SCALE", st->result_scale);
    st->sharp_strength= env_float("RAFT_SHARP",        st->sharp_strength);

    const char *engine_path = env_str("RAFT_ENGINE_PATH",
        "/home/ntc-orin/raft/raft_large_fp16.engine");

    const char *csv_path = env_str("RAFT_CSV_PATH",
        "/home/ntc-orin/raft/output.csv");

    std::fprintf(stderr,
        "[raft_of] Init W=%d H=%d engine=%s\n"
        "[raft_of] ROI=[%.2f,%.2f]x[%.2f,%.2f] step=%d "
        "arrow=%.1f min_mag=%.1f result_scale=%.1f sharp=%.1f\n"
        "[raft_of] px_per_m=%.1f fps=%.1f csv=%s\n",
        W, H, engine_path,
        st->roi_x0, st->roi_x1, st->roi_y0, st->roi_y1,
        st->step, st->arrow_scale, st->min_mag,
        st->result_scale, st->sharp_strength,
        get_px_per_m(), FPS, csv_path);

    if (cudaStreamCreateWithFlags(&st->stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "[raft_of] cudaStreamCreate failed\n");
        st->init_failed = true; return false;
    }

    if (!st->raft.init(engine_path, W, H)) {
        std::fprintf(stderr, "[raft_of] RAFT init failed — will not retry\n");
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

    // Open CSV — write header
    st->csv_file = std::fopen(csv_path, "w");
    if (!st->csv_file) {
        std::fprintf(stderr, "[raft_of] Cannot open CSV: %s\n", csv_path);
        st->init_failed = true; return false;
    }
    std::fprintf(st->csv_file,
        "frame,time_s,"
        "mean_u_px,mean_v_px,"
        "vx_kmh,vy_kmh,"
        "speed_kmh,beta_deg\n");
    std::fflush(st->csv_file);

    st->inited = true;
    std::fprintf(stderr, "[raft_of] Init complete\n");
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
    nv12_to_rgb_fp16(
        egl.d_y, egl.d_uv,
        egl.pitchY, egl.pitchUV,
        W, H, st->d_frame_curr, st->stream);

    // Step 3: sharpen
    preprocess_sharpen(
        st->d_frame_curr, W, H,
        st->sharp_strength, st->stream);

    // Step 4: RAFT inference
    if (st->have_prev) {
        if (!st->raft.infer(st->d_frame_prev, st->d_frame_curr,
                            st->d_flow, st->stream)) {
            std::fprintf(stderr, "[raft_of] infer failed frame %u\n", st->frame_id);
        } else {
            const int rx0=(int)(st->roi_x0*W), rx1=(int)(st->roi_x1*W);
            const int ry0=(int)(st->roi_y0*H), ry1=(int)(st->roi_y1*H);

            // Step 5: reduce flow → mean u,v on GPU
            flow_reduce(
                st->d_flow, W, H,
                rx0, rx1, ry0, ry1,
                st->step, st->d_result, st->stream);

            // Step 6: copy result to host (~8 bytes)
            FlowResult res_h{};
            cudaMemcpyAsync(&res_h, st->d_result, sizeof(FlowResult),
                            cudaMemcpyDeviceToHost, st->stream);
            cudaStreamSynchronize(st->stream);

            // ── Convert to physical units ─────────────────────────────────
            // mean_u: positive = right, negative = forward (camera forward = -x)
            // mean_v: positive = down in image
            // vx = longitudinal speed (forward positive)
            // vy = lateral speed (left positive)
            const float SCALE_TO_KMH = (1.0f / get_px_per_m()) * FPS * 3.6f;
            const float vx_kmh    = -res_h.mean_u * SCALE_TO_KMH;  // forward positive
            const float VY_BIAS_PX = env_float("RAFT_VY_BIAS_PX", 0.0f);
            const float vy_kmh    = (res_h.mean_v - VY_BIAS_PX) * SCALE_TO_KMH;  // lateral
            const float speed_kmh = sqrtf(vx_kmh*vx_kmh + vy_kmh*vy_kmh);
            const float beta_deg  = atan2f(vy_kmh, vx_kmh) * (180.0f / 3.14159265f);
            const float time_s    = (float)st->frame_id / FPS;

            // stdout
            std::fprintf(stdout,
                "frame=%u t=%.3fs  "
                "u=%.3f v=%.3f px/frame  "
                "vx=%.2f vy=%.2f speed=%.2f km/h  beta=%.2f deg\n",
                st->frame_id, time_s,
                res_h.mean_u, res_h.mean_v,
                vx_kmh, vy_kmh, speed_kmh, beta_deg);
            std::fflush(stdout);

            // CSV
            if (st->csv_file) {
                std::fprintf(st->csv_file,
                    "%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    st->frame_id, time_s,
                    res_h.mean_u, res_h.mean_v,
                    vx_kmh, vy_kmh, speed_kmh, beta_deg);
                std::fflush(st->csv_file);
            }

            // Step 7: draw green flow field
            overlay_draw_flow(
                egl.d_y, egl.d_uv,
                egl.pitchY, egl.pitchUV,
                W, H, st->d_flow,
                st->roi_x0, st->roi_x1,
                st->roi_y0, st->roi_y1,
                st->step, st->arrow_scale, st->min_mag,
                st->stream);

            // Step 8: draw blue resultant at center of ROI
            overlay_draw_resultant(
                egl.d_y, egl.d_uv,
                egl.pitchY, egl.pitchUV,
                W, H,
                res_h.mean_u, res_h.mean_v,
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
    if (st->csv_file)     { std::fclose(st->csv_file);    st->csv_file=nullptr; }
    if (st->d_frame_prev) { cudaFree(st->d_frame_prev);   st->d_frame_prev=nullptr; }
    if (st->d_frame_curr) { cudaFree(st->d_frame_curr);   st->d_frame_curr=nullptr; }
    if (st->d_flow)       { cudaFree(st->d_flow);         st->d_flow=nullptr; }
    if (st->d_result)     { cudaFree(st->d_result);       st->d_result=nullptr; }
    if (st->stream)       { cudaStreamDestroy(st->stream); st->stream=nullptr; }
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