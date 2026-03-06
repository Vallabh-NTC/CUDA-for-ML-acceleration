// egl_map.cu
// Maps EGLImageKHR → CUDA pointer (zero-copy).
// NVDEC writes NV12 into NVMM. We just ask CUDA to give us a pointer
// to that same memory — no transfer happens.

#include "egl_map.hpp"

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

static bool s_cuInited = false;

static bool ensure_cu_init()
{
    if (s_cuInited) return true;
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "[egl_map] cuInit failed\n");
        return false;
    }
    s_cuInited = true;
    return true;
}

bool egl_map(EGLImageKHR eglImage, int W, int H, EGLMapResult &out)
{
    out = {};

    if (!eglImage || W <= 0 || H <= 0) return false;
    if (!ensure_cu_init()) return false;

    CUgraphicsResource cuRes = nullptr;

    // Register the EGLImage as a CUDA graphics resource.
    // CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = read + write access.
    if (cuGraphicsEGLRegisterImage(
            &cuRes,
            eglImage,
            CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS)
    {
        std::fprintf(stderr, "[egl_map] cuGraphicsEGLRegisterImage failed\n");
        return false;
    }

    // Get the EGL frame descriptor — this gives us the actual GPU pointers.
    CUeglFrame frame;
    if (cuGraphicsResourceGetMappedEglFrame(&frame, cuRes, 0, 0) != CUDA_SUCCESS)
    {
        std::fprintf(stderr, "[egl_map] cuGraphicsResourceGetMappedEglFrame failed\n");
        cuGraphicsUnregisterResource(cuRes);
        return false;
    }

    // We only handle pitch-linear NV12 (what nvv4l2decoder produces on Orin).
    if (frame.frameType != CU_EGL_FRAME_TYPE_PITCH)
    {
        std::fprintf(stderr, "[egl_map] Unsupported EGL frame type (expected PITCH)\n");
        cuGraphicsUnregisterResource(cuRes);
        return false;
    }

    // NV12 layout:
    //   plane 0 → Y  (luma),  H   rows of W bytes
    //   plane 1 → UV (chroma) H/2 rows of W bytes (U and V interleaved)
    out.d_y    = reinterpret_cast<uint8_t*>(frame.frame.pPitch[0]);
    out.d_uv   = reinterpret_cast<uint8_t*>(frame.frame.pPitch[1]);
    out.pitchY  = static_cast<int>(frame.pitch);
    out.pitchUV = static_cast<int>(frame.pitch);
    out.W      = W;
    out.H      = H;
    out.cuRes  = reinterpret_cast<void*>(cuRes);

    return true;
}

void egl_unmap(EGLMapResult &res)
{
    if (!res.cuRes) return;
    cuGraphicsUnregisterResource(
        reinterpret_cast<CUgraphicsResource>(res.cuRes));
    res.cuRes = nullptr;
    res.d_y   = nullptr;
    res.d_uv  = nullptr;
}