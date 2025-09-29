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
#include "wire_lineedge.cuh"

namespace {

struct State {
    CUcontext    ctx    = nullptr;
    cudaStream_t stream = nullptr;

    int W = 0, H = 0;

    struct MaskEntry {
        CUdeviceptr dMask = 0;
        size_t      pMask = 0;
    };
    std::vector<MaskEntry> masks;
    bool masks_loaded = false;

    // fisheye center + radial offsets (pixels)
    float cx = -1.f, cy = -1.f;
    float offIn = 20.f, offOut = 20.f;

    // debug colors (optional)
    uint8_t y_mask = 235, u_mask = 128, v_mask = 128;
};

static std::once_flag g_once;
static CUcontext g_primary = nullptr;
static CUdevice  g_dev = 0;

static void retain_primary_once() {
    cuInit(0);
    cuDeviceGet(&g_dev, 0);
    cuDevicePrimaryCtxRetain(&g_primary, g_dev);
    cuCtxSetCurrent(g_primary);
}

// ---------- Host I/O ----------
static bool read_global_meta(const char* metaPath, int& W, int& H, int& count)
{
    std::ifstream m(metaPath);
    if (!m) return false;
    W = H = 0; count = -1;
    m >> W >> H;
    if (!m.fail()) { int tmp; if (m >> tmp) count = tmp; }
    return (W > 0 && H > 0);
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary); return (bool)f;
}

static bool load_mask_raw_only(const char* rawPath, int W, int H, std::vector<uint8_t>& buf)
{
    std::ifstream r(rawPath, std::ios::binary);
    if (!r) { fprintf(stderr, "[wire] cannot open %s\n", rawPath); return false; }
    r.seekg(0, std::ios::end);
    std::streamsize sz = r.tellg();
    r.seekg(0, std::ios::beg);
    const std::streamsize need = (std::streamsize)W * (std::streamsize)H;
    if (sz != need) {
        fprintf(stderr, "[wire] %s size mismatch: got %lld, need %lld\n",
                rawPath, (long long)sz, (long long)need);
        return false;
    }
    buf.resize((size_t)need);
    r.read(reinterpret_cast<char*>(buf.data()), need);
    return (r.gcount() == need);
}

// ---------- Clean up ----------
static void free_masks(State* s){
    for (auto& m : s->masks) if (m.dMask) cuMemFree(m.dMask);
    s->masks.clear();
    s->masks_loaded = false;
}

static void free_all(State* s){
    if (!s) return;
    cuCtxSetCurrent(s->ctx);
    free_masks(s);
    if (s->stream) cudaStreamDestroy(s->stream);
}

// ---------- Load masks for current WxH ----------
static void ensure_masks(State* st, int frameW, int frameH)
{
    if (st->masks_loaded && st->W==frameW && st->H==frameH) return;

    free_masks(st);

    int metaW=0, metaH=0, metaCount=-1;
    const char* META = "/dev/shm/wire_mask.meta";
    bool haveMeta = read_global_meta(META, metaW, metaH, metaCount);

    int W = frameW, H = frameH;
    if (haveMeta && (metaW!=frameW || metaH!=frameH)) {
        fprintf(stderr, "[wire] WARNING: meta WxH (%d x %d) != frame WxH (%d x %d). Using frame WxH.\n",
                metaW, metaH, frameW, frameH);
    }

    const int MAX_MASKS = 256;
    int idxEnd = (metaCount > 0) ? metaCount : 999;
    int loaded = 0;

    for (int idx = 1; idx <= idxEnd && loaded < MAX_MASKS; ++idx) {
        char base[64];
        snprintf(base, sizeof(base), "/dev/shm/wire_mask_%03d", idx);
        std::string raw = std::string(base) + ".raw";
        if (!file_exists(raw)) {
            if (metaCount > 0) fprintf(stderr, "[wire] expected %s not found; skip\n", raw.c_str());
            continue;
        }

        std::vector<uint8_t> hostMask;
        if (!load_mask_raw_only(raw.c_str(), W, H, hostMask)) {
            fprintf(stderr, "[wire] skip %s (size/read error)\n", raw.c_str());
            continue;
        }

        State::MaskEntry me{};
        size_t pitchBytes = 0;
        CUresult rc = cuMemAllocPitch(&me.dMask, &pitchBytes, (size_t)W, (size_t)H, 4);
        if (rc != CUDA_SUCCESS) { fprintf(stderr, "[wire] cuMemAllocPitch failed for %s\n", raw.c_str()); continue; }

        CUDA_MEMCPY2D c{};
        c.srcMemoryType = CU_MEMORYTYPE_HOST;
        c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        c.srcHost       = hostMask.data();
        c.srcPitch      = W;
        c.dstDevice     = me.dMask;
        c.dstPitch      = pitchBytes;
        c.WidthInBytes  = W;
        c.Height        = H;
        rc = cuMemcpy2D(&c);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "[wire] cuMemcpy2D(%s) failed\n", raw.c_str());
            cuMemFree(me.dMask);
            continue;
        }

        me.pMask = pitchBytes;
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

// ---------- Lifecycle ----------
static State* create_state(){
    std::call_once(g_once, retain_primary_once);
    auto* s = new State();
    s->ctx = g_primary; cuCtxSetCurrent(s->ctx);

#if CUDART_VERSION >= 11000
    cudaStreamCreateWithPriority(&s->stream, cudaStreamNonBlocking, 0);
#else
    cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking);
#endif

    if (const char* v = std::getenv("WIRE_MASK_Y")) s->y_mask = (uint8_t)atoi(v);
    if (const char* v = std::getenv("WIRE_MASK_U")) s->u_mask = (uint8_t)atoi(v);
    if (const char* v = std::getenv("WIRE_MASK_V")) s->v_mask = (uint8_t)atoi(v);

    if (const char* v = std::getenv("WIRE_RAD_IN"))  s->offIn  = std::max(1, atoi(v));
    if (const char* v = std::getenv("WIRE_RAD_OUT")) s->offOut = std::max(1, atoi(v));

    // Center defaults to image center; can override via env in pixels
    s->cx = -1.f; s->cy = -1.f;
    if (const char* v = std::getenv("WIRE_CX")) s->cx = (float)atof(v);
    if (const char* v = std::getenv("WIRE_CY")) s->cy = (float)atof(v);

    fprintf(stderr, "[wire] radial (fisheye-aligned) disappearance. offIn=%.1f offOut=%.1f\n",
            s->offIn, s->offOut);
    return s;
}

static void destroy_state(State* s){
    free_all(s);
    delete s;
}

static void pre_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}
static void post_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}

// ---------- Main GPU path ----------
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
    const int pitch = (int)f.pitch;

    ensure_masks(st, W, H);
    if (!st->masks_loaded) { cuGraphicsUnregisterResource(res); return; }

    // Fisheye center defaults to image center if not set
    float cx = (st->cx >= 0.f) ? st->cx : 0.5f * (float)W;
    float cy = (st->cy >= 0.f) ? st->cy : 0.5f * (float)H;

    // Per-mask radial disappearance
    for (auto& me : st->masks) {
        wire::disappear_mask_radial_nv12(
            dY,  pitch,
            dUV, pitch,
            W, H,
            (const uint8_t*)(uintptr_t)me.dMask, (int)me.pMask,
            cx, cy,
            st->offIn, st->offOut,
            st->stream);
    }

    cudaStreamSynchronize(st->stream);
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
    // Nothing: resources are freed when pipeline tears down.
}
