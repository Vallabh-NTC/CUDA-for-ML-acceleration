// egl_copy.cu
// CUDA-EGL interop helper: copy NV12 Y plane from EGLImage into CUDA pitched buffer.
// No printf/device logging.

#include <cstdint>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

#include "egl_copy.hpp"

static inline bool cu_ok(CUresult r) { return r == CUDA_SUCCESS; }
static inline bool rt_ok(cudaError_t e) { return e == cudaSuccess; }

__global__ void copy_pitch_kernel(const uint8_t *src, int srcPitch,
                                  uint8_t *dst, int dstPitch,
                                  int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // pixel x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // pixel y
    if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return;

    dst[y * dstPitch + x] = src[y * srcPitch + x];
}

extern "C" void egl_nv12_copy_y_to_cuda(EGLImageKHR eglImage,
                                       uint8_t *dstY, int dstPitch,
                                       int W, int H)
{
    if (!eglImage || !dstY || W <= 0 || H <= 0 || dstPitch <= 0) return;

    static bool cuInitDone = false;
    if (!cuInitDone) {
        if (!cu_ok(cuInit(0))) return;
        cuInitDone = true;
    }

    static cudaStream_t stream = nullptr;
    if (!stream) {
        if (!rt_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)))
            return;
    }

    CUgraphicsResource cuRes = nullptr;
    CUeglFrame eglFrame;

    if (!cu_ok(cuGraphicsEGLRegisterImage(&cuRes, (EGLImageKHR)eglImage,
                                          CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)))
        return;

    if (!cu_ok(cuGraphicsResourceGetMappedEglFrame(&eglFrame, cuRes, 0, 0))) {
        cuGraphicsUnregisterResource(cuRes);
        return;
    }

    // NV12: plane 0 is Y (8-bit)
    if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
        const uint8_t *srcY = (const uint8_t*)eglFrame.frame.pPitch[0];
        int srcPitch = (int)eglFrame.pitch;

        dim3 block(32, 8);
        dim3 grid((W + block.x - 1) / block.x,
                  (H + block.y - 1) / block.y);

        copy_pitch_kernel<<<grid, block, 0, stream>>>(srcY, srcPitch, dstY, dstPitch, W, H);
        cudaGetLastError(); // swallow
        cudaStreamSynchronize(stream);
    } else if (eglFrame.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
        // Y plane is an array
        CUarray cuArr = eglFrame.frame.pArray[0];

        // Use cudaMemcpy2DFromArray into pitched dst
        // Copy width=W bytes per row for H rows.
        cudaMemcpy2DFromArrayAsync(dstY, (size_t)dstPitch,
                                   (cudaArray_t)cuArr,
                                   0, 0,
                                   (size_t)W, (size_t)H,
                                   cudaMemcpyDeviceToDevice,
                                   stream);
        cudaGetLastError();
        cudaStreamSynchronize(stream);
    }

    cuGraphicsUnregisterResource(cuRes);
}
