#include <cstdio>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>

#include "nvivafilter_customer_api.hpp"
#include "color_ops.cuh"
#include "runtime_controls.hpp"

namespace {
struct State {
    CUcontext     ctx   = nullptr;   // primary
    cudaStream_t  stream= nullptr;
    pe::RuntimeControls controls{"/home/jetson_ntc/editor.json"};
};
static State g; 

static void ensure_cuda_once() {
    static bool inited = false;
    if (inited) return;
    inited = true;
    cuInit(0);
    CUdevice dev=0; cuDeviceGet(&dev, 0);
    CUcontext primary=nullptr; cuDevicePrimaryCtxRetain(&primary, dev);
    g.ctx = primary;
#if CUDART_VERSION >= 11000
    cudaStreamCreateWithPriority(&g.stream, cudaStreamNonBlocking, 0);
#else
    cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking);
#endif
    fprintf(stderr, "[pe] post-editing initialized (single instance)\n");
}
} // anon

static void pre_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}
static void post_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}

static void gpu_process(EGLImageKHR image, void **)
{
    ensure_cuda_once();
    cuCtxSetCurrent(g.ctx);

    //NV12
    CUgraphicsResource res=nullptr;
    if(cuGraphicsEGLRegisterImage(&res,image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)!=CUDA_SUCCESS) return;
    CUeglFrame f{}; if(cuGraphicsResourceGetMappedEglFrame(&f,res,0,0)!=CUDA_SUCCESS){ cuGraphicsUnregisterResource(res); return; }
    if(f.frameType!=CU_EGL_FRAME_TYPE_PITCH || f.planeCount<2){ cuGraphicsUnregisterResource(res); return; }

    uint8_t* dY  = static_cast<uint8_t*>(f.frame.pPitch[0]);
    uint8_t* dUV = static_cast<uint8_t*>(f.frame.pPitch[1]);
    const int W = (int)f.width, H=(int)f.height, pitch=(int)f.pitch;

    
    icp::ColorParams cp = g.controls.current();

    // Kernel
    icp::launch_tone_saturation_nv12(
        dY, W, H, pitch,
        dUV,     pitch,
        cp,
        g.stream);

    cudaStreamSynchronize(g.stream);
    cuGraphicsUnregisterResource(res);
}

extern "C" void init(CustomerFunction* f)
{
    if(!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;
    ensure_cuda_once();
}
extern "C" void deinit(void)
{
    cuCtxSetCurrent(g.ctx);
    if (g.stream) { cudaStreamDestroy(g.stream); g.stream=nullptr; }
    if (g.ctx)    { CUdevice dev=0; cuCtxGetDevice(&dev); cuDevicePrimaryCtxRelease(dev); g.ctx=nullptr; }
}
