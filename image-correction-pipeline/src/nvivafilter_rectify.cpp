// image-correction-pipeline/src/nvivafilter_rectify.cu
// -----------------------------------------------------------------------------
// This file is the "glue" between GStreamer’s gst-nvivafilter and CUDA
// rectification kernel. It maps the incoming NVMM EGLImage to a CUDA-accessible
// frame (CUeglFrame), checks that it is pitch-linear ABGR, and then calls the
// kernel launcher. Runtime tunables are pulled from a JSON file via the
// RuntimeControls helper.
//
// LIFECYCLE (gst-nvivafilter):
//   init()         → called once when the .so is loaded; we assign our callbacks
//   fPreProcess    → called when the pipeline goes to PLAYING; we allocate ctx
//   fGPUProcess    → called for each input EGLImage buffer; we run the kernel
//   fPostProcess   → called when the pipeline stops; we clean up
//
// THREADING:
//   - GStreamer calls fGPUProcess() on the streaming thread.GPU work is run on a CUDA stream 
//     and synchronized before returning, ensuring the frame is ready when passed downstream.
//   - RuntimeControls runs a separate, tiny thread to hot-reload JSON.
//
// FORMAT ASSUMPTION:
//   - We hard-require CU_EGL_COLOR_FORMAT_ABGR (i.e., RGBA pixels in ABGR enum).
//   - If your camera is UYVY/YUY2/NV12, add: nvvidconv ! video/x-raw(memory:NVMM),format=RGBA
//     *before* the nvivafilter in your pipeline.
//
// DEPENDENCY:
//   - "customer_functions.h" comes from Jetson Multimedia API (in targetfs).
// -----------------------------------------------------------------------------
// image-correction-pipeline/src/nvivafilter_rectify.cu
// -----------------------------------------------------------------------------
// Glue between gst-nvivafilter and the CUDA rectify kernel.
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

#include "nvivafilter_customer_api.hpp" // <-- local ABI mirror (no external dep)
#include "runtime_controls.hpp"
#include "rectify_kernels.cuh"

#define CUCHK(x) do { CUresult e=(x); if(e!=CUDA_SUCCESS){ \
  const char* s=nullptr; cuGetErrorString(e,&s); \
  std::fprintf(stderr,"CUDA-DRIVER error %d (%s) at %s:%d\n",(int)e,s?s:"?",__FILE__,__LINE__); }} while(0)
#define CUDACHK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA-RUNTIME error %s at %s:%d\n", cudaGetErrorString(e),__FILE__,__LINE__); }} while(0)

struct FilterCtx {
    RuntimeControls* controls = nullptr;
    cudaStream_t     stream   = nullptr;
};

static void pre_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*,
                        ColorFormat*, unsigned int, void **usrptr)
{
    // Ensure CUDA driver is ready (safe to call multiple times)
    CUCHK(cuInit(0));

    auto* ctx = new FilterCtx();
    CUDACHK(cudaStreamCreate(&ctx->stream));

    const char* envp = std::getenv("RECTIFY_CONTROLS_JSON");
    std::string path = (envp && *envp) ? std::string(envp) : "/opt/rectify/controls.json";
    ctx->controls = new RuntimeControls(path);

    *usrptr = ctx;
    std::fprintf(stderr, "[nvivafilter_rectify] pre_process: watching %s\n", path.c_str());
}

static void post_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*,
                         ColorFormat*, unsigned int, void **usrptr)
{
    auto* ctx = (FilterCtx*)(*usrptr);
    if (!ctx) return;
    delete ctx->controls;  ctx->controls = nullptr;
    if (ctx->stream) { cudaStreamDestroy(ctx->stream); ctx->stream = nullptr; }
    delete ctx; *usrptr = nullptr;
}

static void gpu_process(EGLImageKHR image, void **usrptr)
{
    auto* ctx = (FilterCtx*)(*usrptr);
    if (!ctx) return;

    CUgraphicsResource res = nullptr;
    CUeglFrame frame{};
    // Register and fetch CUeglFrame
    CUCHK(cuGraphicsEGLRegisterImage(&res, image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE));
    CUCHK(cuGraphicsResourceGetMappedEglFrame(&frame, res, 0, 0));

    // Require pitch-linear ABGR (RGBA byte layout reported as ABGR enum)
    if (frame.frameType != CU_EGL_FRAME_TYPE_PITCH ||
        frame.eglColorFormat != CU_EGL_COLOR_FORMAT_ABGR)
    {
        std::fprintf(stderr, "[nvivafilter_rectify] Unsupported frame (type=%d col=%d). "
                             "Insert nvvidconv to RGBA before nvivafilter.\n",
                     frame.frameType, frame.eglColorFormat);
        CUCHK(cuGraphicsUnregisterResource(res));
        return;
    }

    // In-place ABGR processing
    uint8_t* d_ptr = (uint8_t*)frame.frame.pPitch[0];
    int pitch = frame.pitch;
    int w = frame.width;
    int h = frame.height;

    RectifyConfig cfg = ctx->controls->get();
    launch_rectify_kernel(d_ptr, w, h, pitch, d_ptr, w, h, pitch, cfg, ctx->stream);

    CUDACHK(cudaStreamSynchronize(ctx->stream));
    CUCHK(cuGraphicsUnregisterResource(res));
}

extern "C" void init(CustomerFunction* f) {
    if (!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;
}

extern "C" void deinit(void) {}
