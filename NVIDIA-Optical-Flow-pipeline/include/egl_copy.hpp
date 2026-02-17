// include/egl_copy.hpp
#pragma once
#include <cstdint>

// Forward typedef to avoid pulling EGL headers in CUDA TU headers.
typedef void* EGLImageKHR;

#ifdef __cplusplus
extern "C" {
#endif

// Copy NV12 luma (Y plane) from an EGLImage into a CUDA pitched buffer.
// dstY is a device pointer, dstPitch in bytes.
// W,H are in pixels.
void egl_nv12_copy_y_to_cuda(EGLImageKHR eglImage,
                             uint8_t *dstY, int dstPitch,
                             int W, int H);

#ifdef __cplusplus
}
#endif
