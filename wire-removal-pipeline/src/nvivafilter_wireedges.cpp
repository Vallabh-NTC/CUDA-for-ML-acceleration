// nvivafilter_wireline.cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>
#include <algorithm>

#include "nvivafilter_customer_api.hpp"
#include "wire_lineedge.cuh"   // declarations: top_bottom_from_mask, disappear_band_nv12, overlays

namespace {

struct State {
    // CUDA context & stream
    CUcontext    ctx    = nullptr;
    cudaStream_t stream = nullptr;

    // Frame geometry we were prepared for
    int W = 0, H = 0;

    // === Multiple masks ===
    struct MaskEntry {
        CUdeviceptr dMask = 0;  // device mask (pitched)
        size_t      pMask = 0;  // mask pitch in bytes
        int*        dTop  = nullptr; // per-column top row
        int*        dBot  = nullptr; // per-column bottom row
        bool        band_ready = false;
    };
    std::vector<MaskEntry> masks;  // all loaded masks for this WxH
    bool        masks_loaded = false;

    // NV12 overlay colors (debug)
    uint8_t y_mask = 235, u_mask = 128, v_mask = 128; // mask band (white)
    uint8_t y_top  = 80,  u_top  = 16,  v_top  = 146; // top polyline
    uint8_t y_bot  = 210, u_bot  = 16,  v_bot  = 16;  // bottom polyline

    // Disappear offsets (can be overridden via env)
    int offTop = 3; // rows above mask to copy from
    int offBot = 3; // rows below mask to copy from
};

// One-time retention of primary context (Jetson-style)
static std::once_flag g_once;
static CUcontext g_primary = nullptr;
static CUdevice  g_dev = 0;

static void retain_primary_once() {
    cuInit(0);
    cuDeviceGet(&g_dev, 0);
    cuDevicePrimaryCtxRetain(&g_primary, g_dev);
    cuCtxSetCurrent(g_primary);
}

// ---------- Host I/O helpers ----------

// read one global meta: "W H N"  (N optional)
static bool read_global_meta(const char* metaPath, int& W, int& H, int& count)
{
    std::ifstream m(metaPath);
    if (!m) return false;
    W = H = 0; count = -1;
    m >> W >> H;
    if (!m.fail()) {
        int tmp;
        if (m >> tmp) count = tmp;
    }
    return (W > 0 && H > 0);
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return (bool)f;
}

// raw-only loader, validates size == W*H
static bool load_mask_raw_only(const char* rawPath, int W, int H, std::vector<uint8_t>& buf)
{
    std::ifstream r(rawPath, std::ios::binary);
    if (!r) { fprintf(stderr, "[wire] cannot open %s\n", rawPath); return false; }

    r.seekg(0, std::ios::end);
    std::streamsize sz = r.tellg();
    r.seekg(0, std::ios::beg);
    const std::streamsize need = (std::streamsize)W * (std::streamsize)H;
    if (sz != need) {
        fprintf(stderr, "[wire] %s size mismatch: got %lld, need %lld (W*H)\n",
                rawPath, (long long)sz, (long long)need);
        return false;
    }

    buf.resize((size_t)need);
    r.read(reinterpret_cast<char*>(buf.data()), need);
    if (r.gcount() != need) {
        fprintf(stderr, "[wire] %s read short: %lld/%lld\n",
                rawPath, (long long)r.gcount(), (long long)need);
        return false;
    }
    return true;
}

// ---------- Resource cleanup ----------
static void free_masks(State* s){
    for (auto& m : s->masks) {
        if (m.dMask) cuMemFree(m.dMask);
        if (m.dTop)  cudaFree(m.dTop);
        if (m.dBot)  cudaFree(m.dBot);
    }
    s->masks.clear();
    s->masks_loaded = false;
}

static void free_all(State* s){
    if (!s) return;
    cuCtxSetCurrent(s->ctx);
    free_masks(s);
    if (s->stream) cudaStreamDestroy(s->stream);
}

// ---------- Mask loader: one global .meta + many .raw ----------
static void ensure_masks(State* st, int frameW, int frameH)
{
    // If already loaded for this WxH, keep them
    if (st->masks_loaded && st->W == frameW && st->H == frameH) return;

    // (Re)load all masks referenced by /dev/shm/wire_mask.meta
    free_masks(st);

    int metaW=0, metaH=0, metaCount=-1;
    const char* META = "/dev/shm/wire_mask.meta";
    bool haveMeta = read_global_meta(META, metaW, metaH, metaCount);

    int W = frameW, H = frameH;
    if (haveMeta) {
        if (metaW != frameW || metaH != frameH) {
            fprintf(stderr, "[wire] WARNING: meta WxH (%d x %d) != frame WxH (%d x %d). Using frame WxH; raw must still be W*H.\n",
                    metaW, metaH, frameW, frameH);
        }
    } else {
        fprintf(stderr, "[wire] NOTE: %s not found/invalid; scanning raws and validating sizes against frame WxH.\n", META);
    }

    const int MAX_MASKS = 256;
    int loaded = 0;

    // If metaCount is known, enumerate exactly that many indices; else scan range.
    int idxStart = 1;
    int idxEnd   = (metaCount > 0) ? metaCount : 999;

    for (int idx = idxStart; idx <= idxEnd && loaded < MAX_MASKS; ++idx) {
        char base[64];
        snprintf(base, sizeof(base), "/dev/shm/wire_mask_%03d", idx);
        std::string raw = std::string(base) + ".raw";

        if (!file_exists(raw)) {
            if (metaCount > 0) {
                fprintf(stderr, "[wire] expected %s but not found; skipping\n", raw.c_str());
            }
            continue;
        }

        // Read host data (raw-only; expect W*H bytes)
        std::vector<uint8_t> hostMask;
        if (!load_mask_raw_only(raw.c_str(), W, H, hostMask)) {
            fprintf(stderr, "[wire] skip %s: raw load failed\n", raw.c_str());
            continue;
        }

        // Upload pitched device mask (elemSize=4 for cuMemAllocPitch)
        State::MaskEntry me{};
        size_t pitchBytes = 0;
        CUresult rc = cuMemAllocPitch(&me.dMask, &pitchBytes, (size_t)W, (size_t)H, 4);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "[wire] cuMemAllocPitch failed for %s (%d)\n", raw.c_str(), (int)rc);
            continue;
        }

        CUDA_MEMCPY2D c{};
        c.srcMemoryType = CU_MEMORYTYPE_HOST;
        c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        c.srcHost       = hostMask.data();
        c.srcPitch      = W;          // tightly packed host
        c.dstDevice     = me.dMask;
        c.dstPitch      = pitchBytes; // pitched device
        c.WidthInBytes  = W;
        c.Height        = H;
        rc = cuMemcpy2D(&c);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "[wire] cuMemcpy2D(mask %s) failed (%d)\n", raw.c_str(), (int)rc);
            cuMemFree(me.dMask);
            continue;
        }

        // Per-mask top/bottom buffers
        cudaMalloc(&me.dTop, W * sizeof(int));
        cudaMalloc(&me.dBot, W * sizeof(int));

        me.pMask      = pitchBytes;
        me.band_ready = false;

        st->masks.push_back(me);
        ++loaded;
        fprintf(stderr, "[wire] loaded mask %s (pitch=%zu)\n", raw.c_str(), me.pMask);
    }

    if (loaded == 0) {
        fprintf(stderr, "[wire] no masks found; frames will pass through.\n");
        st->masks_loaded = false;
        return;
    }

    st->W = W; st->H = H;
    st->masks_loaded = true;
    fprintf(stderr, "[wire] loaded %d mask(s) @ %dx%d\n", loaded, W, H);
}

// ---------- Boilerplate init/destroy ----------
static State* create_state(){
    std::call_once(g_once, retain_primary_once);
    auto* s = new State();
    s->ctx = g_primary; cuCtxSetCurrent(s->ctx);

#if CUDART_VERSION >= 11000
    cudaStreamCreateWithPriority(&s->stream, cudaStreamNonBlocking, 0);
#else
    cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking);
#endif

    // Optional env overrides
    if (const char* v = std::getenv("WIRE_MASK_Y")) s->y_mask = (uint8_t)atoi(v);
    if (const char* v = std::getenv("WIRE_MASK_U")) s->u_mask = (uint8_t)atoi(v);
    if (const char* v = std::getenv("WIRE_MASK_V")) s->v_mask = (uint8_t)atoi(v);

    if (const char* v = std::getenv("WIRE_TOP_Y")) s->y_top  = (uint8_t)atoi(v);
    if (const char* v = std::getenv("WIRE_TOP_U")) s->u_top  = (uint8_t)atoi(v);
    if (const char* v = std::getenv("WIRE_TOP_V")) s->v_top  = (uint8_t)atoi(v);

    if (const char* v = std::getenv("WIRE_BOT_Y")) s->y_bot  = (uint8_t)atoi(v);
    if (const char* v = std::getenv("WIRE_BOT_U")) s->u_bot  = (uint8_t)atoi(v);
    if (const char* v = std::getenv("WIRE_BOT_V")) s->v_bot  = (uint8_t)atoi(v);

    if (const char* v = std::getenv("WIRE_OFF_TOP")) s->offTop = std::max(1, atoi(v));
    if (const char* v = std::getenv("WIRE_OFF_BOT")) s->offBot = std::max(1, atoi(v));

    fprintf(stderr, "[wire] copy-from-neighbors mode (multi-mask). offTop=%d offBot=%d\n",
            s->offTop, s->offBot);
    return s;
}

static void destroy_state(State* s){
    free_all(s);
    delete s;
}

static void pre_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}
static void post_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}

// ---------- Main GPU process ----------
static void gpu_process(EGLImageKHR image, void **userPtr)
{
    State* st = static_cast<State*>(*userPtr);
    if (!st) { st = create_state(); *userPtr = st; }
    cuCtxSetCurrent(st->ctx);

    CUgraphicsResource res = nullptr;
    if (cuGraphicsEGLRegisterImage(&res, image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) return;

    CUeglFrame f{};
    if (cuGraphicsResourceGetMappedEglFrame(&f, res, 0, 0) != CUDA_SUCCESS) {
        cuGraphicsUnregisterResource(res); return;
    }
    if (f.frameType != CU_EGL_FRAME_TYPE_PITCH || f.planeCount < 2) {
        cuGraphicsUnregisterResource(res); return;
    }

    uint8_t* dY  = static_cast<uint8_t*>(f.frame.pPitch[0]);
    uint8_t* dUV = static_cast<uint8_t*>(f.frame.pPitch[1]);
    const int W  = (int)f.width;
    const int H  = (int)f.height;
    const int pitch = (int)f.pitch;  // bytes/row (Y; UV shares this on Jetson)

    // Load all masks that match this resolution (if not already)
    ensure_masks(st, W, H);

    if (st->masks_loaded) {
        // Compute bands (top/bottom) once per mask
        bool need_sync = false;
        for (auto& me : st->masks) {
            if (!me.band_ready) {
                wire::top_bottom_from_mask(
                    (const uint8_t*)(uintptr_t)me.dMask, (int)me.pMask,
                    W, H, me.dTop, me.dBot, st->stream);
                me.band_ready = true;
                need_sync = true;
            }
        }
        if (need_sync) {
            cudaStreamSynchronize(st->stream);
            fprintf(stderr, "[wire] bands computed for %zu mask(s).\n", st->masks.size());
        }

        // Apply disappearance sequentially for each mask
        for (auto& me : st->masks) {
            wire::disappear_band_nv12(
                dY,  pitch,
                dUV, pitch,
                W, H,
                me.dTop, me.dBot,
                st->offTop, st->offBot, // configurable donor offsets
                0.f, 0.f,
                st->stream);
        }

        // Keep output deterministic for encoder
        cudaStreamSynchronize(st->stream);

        // --- Optional debug overlays (comment out for production) ---
        // for (auto& me : st->masks) {
        //     wire::overlay_polyline_nv12(dY, pitch, dUV, pitch, W, H,
        //                                 me.dTop, st->y_top, st->u_top, st->v_top, st->stream);
        //     wire::overlay_polyline_nv12(dY, pitch, dUV, pitch, W, H,
        //                                 me.dBot, st->y_bot, st->u_bot, st->v_bot, st->stream);
        //     wire::overlay_mask_nv12(dY, W, H, pitch, dUV, pitch,
        //         (const uint8_t*)(uintptr_t)me.dMask, (int)me.pMask,
        //         st->y_mask, st->u_mask, st->v_mask, st->stream);
        // }
        // cudaStreamSynchronize(st->stream);
    }

    cuGraphicsUnregisterResource(res);
}

} // anon

extern "C" void init(CustomerFunction* f){
    if (!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;
}

extern "C" void deinit(void){
    // Freed on pipeline teardown via nvivafilter; nothing to do here.
}
