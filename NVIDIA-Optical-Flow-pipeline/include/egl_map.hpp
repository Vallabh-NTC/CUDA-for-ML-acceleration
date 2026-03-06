#pragma once
// egl_map.hpp
// Maps an EGLImageKHR (NVMM) to a CUDA pointer without copying any bytes.
// The returned pointer points directly into the NVMM buffer where NVDEC
// wrote the NV12 frame — zero CPU involvement.

#include <cstdint>
#include <EGL/egl.h>
#include <EGL/eglext.h>

struct EGLMapResult
{
    uint8_t *d_y;        // Y  plane — [H * pitchY] bytes
    uint8_t *d_uv;       // UV plane — [H/2 * pitchUV] bytes (interleaved)
    int      pitchY;     // stride in bytes of Y  plane
    int      pitchUV;    // stride in bytes of UV plane
    int      W;          // frame width  in pixels
    int      H;          // frame height in pixels
    void    *cuRes;      // opaque CUgraphicsResource — must be passed to egl_unmap()
};

// Map eglImage as CUDA resource.
bool egl_map(EGLImageKHR eglImage, int W, int H, EGLMapResult &out);

// Unmap and release the CUDA graphics resource.
void egl_unmap(EGLMapResult &res);