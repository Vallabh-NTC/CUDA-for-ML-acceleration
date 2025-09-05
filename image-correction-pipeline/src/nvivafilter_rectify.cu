// image-correction-pipeline/src/nvivafilter_rectify.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <nvbufsurface.h>  // NVMM surface type

#include <cstdio>
#include <cstring>

#include "image_correction.hpp"   // <-- your RectifyConfig + fisheye_rectify_rgba()

// ---- NVMM <-> CUDA helpers ----
static bool map_nvmm_to_cuda(NvBufSurface* surf, int idx,
                             CUeglFrame& outFrame,
                             cudaGraphicsResource_t& outRes)
{
    if (!surf || idx < 0 || idx >= surf->numFilled) return false;

    if (NvBufSurfaceMapEglImage(surf, idx) != 0) return false;
    EGLImageKHR eglImage = surf->surfaceList[idx].mappedAddr.eglImage;
    if (!eglImage) { NvBufSurfaceUnMapEglImage(surf, idx); return false; }

    if (cudaGraphicsEGLRegisterImage(&outRes, eglImage, cudaGraphicsRegisterFlagsNone) != cudaSuccess) {
        NvBufSurfaceUnMapEglImage(surf, idx);
        return false;
    }
    if (cuGraphicsResourceGetMappedEglFrame(&outFrame, outRes, 0, 0) != CUDA_SUCCESS) {
        cudaGraphicsUnregisterResource(outRes);
        NvBufSurfaceUnMapEglImage(surf, idx);
        return false;
    }
    return true;
}

static void unmap_nvmm_from_cuda(NvBufSurface* surf, int idx,
                                 cudaGraphicsResource_t res)
{
    cudaGraphicsUnregisterResource(res);
    NvBufSurfaceUnMapEglImage(surf, idx);
}

// ---- nvivafilter entrypoints ----
// (nvivafilter looks for these symbols)
extern "C" void pre_process() { /* optional one-time init */ }
extern "C" void post_process() { /* optional cleanup */ }

extern "C" void gpu_process(NvBufSurface* in_surf,
                            NvBufSurface* out_surf,
                            CUstream stream)
{
    const int idx = 0;  // one buffer per call

    // Map input + output NVMM surfaces to CUDA device pointers (via EGL)
    CUeglFrame inF{}, outF{};
    cudaGraphicsResource_t inRes=nullptr, outRes=nullptr;

    if (!map_nvmm_to_cuda(in_surf,  idx, inF,  inRes))  return;
    if (!map_nvmm_to_cuda(out_surf, idx, outF, outRes)) { unmap_nvmm_from_cuda(in_surf, idx, inRes); return; }

    // Expect RGBA pitch-linear frames on both sides.
    // On Jetson, CUDA reports RGBA as CU_EGL_COLOR_FORMAT_ABGR (byte order compatible).
    bool ok =
        (inF.frameType  == CU_EGL_FRAME_TYPE_PITCH) &&
        (outF.frameType == CU_EGL_FRAME_TYPE_PITCH) &&
        (inF.eglColorFormat  == CU_EGL_COLOR_FORMAT_ABGR) &&
        (outF.eglColorFormat == CU_EGL_COLOR_FORMAT_ABGR);

    if (!ok) {
        unmap_nvmm_from_cuda(out_surf, idx, outRes);
        unmap_nvmm_from_cuda(in_surf,  idx, inRes);
        return;
    }

    // Device pointers + pitches (bytes per row)
    const uint8_t* d_src = static_cast<const uint8_t*>(inF.frame.pPitch[0]);
    uint8_t*       d_dst = static_cast<uint8_t*>(outF.frame.pPitch[0]);

    const int src_w = inF.width;
    const int src_h = inF.height;
    const int dst_w = outF.width;
    const int dst_h = outF.height;
    const size_t src_stride = static_cast<size_t>(inF.pitch);
    const size_t dst_stride = static_cast<size_t>(outF.pitch);

    // Use your existing rectification config (same values as in main.cu)
    icp::RectifyConfig cfg{};
    cfg.fish_fov_deg = 195.1f;
    cfg.out_hfov_deg = 90.0f;
    cfg.cx_f = 959.50f;
    cfg.cy_f = 539.50f;
    cfg.r_f  = 1100.77f;
    cfg.out_width = dst_w;

    // Default color controls (you can wire runtime controls later)
    cfg.brightness = 0.0f;
    cfg.contrast   = 1.0f;
    cfg.saturation = 1.0f;
    cfg.gamma      = 1.0f;
    cfg.wb_r = 1.0f; cfg.wb_g = 1.0f; cfg.wb_b = 1.0f;

    // Call your CUDA implementation directly on NVMM device memory
    icp::fisheye_rectify_rgba(
        d_src, src_w, src_h, src_stride,
        d_dst, dst_w, dst_h, dst_stride,
        cfg,
        stream
    );

    // Ensure completion before releasing the surfaces
    cudaStreamSynchronize(stream);

    unmap_nvmm_from_cuda(out_surf, idx, outRes);
    unmap_nvmm_from_cuda(in_surf,  idx, inRes);
}

// Some builds of nvivafilter request an init() that provides the function pointers.
struct CustomerFunction {
    void (*fPreProcess)();
    void (*fGPUProcess)(NvBufSurface*, NvBufSurface*, CUstream);
    void (*fPostProcess)();
};
extern "C" void init(CustomerFunction* f) {
    if (!f) return;
    f->fPreProcess  = &pre_process;
    f->fGPUProcess  = &gpu_process;
    f->fPostProcess = &post_process;
}
