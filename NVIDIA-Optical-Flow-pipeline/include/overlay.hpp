#pragma once
#include <cstdint>

// Forward typedef to avoid pulling EGL headers in CUDA TU headers.
typedef void* EGLImageKHR;

struct Arrow
{
    int x0, y0, x1, y1;
};

#ifdef __cplusplus
extern "C" {
#endif

// Draw arrows on NV12 in-place via CUDA-EGL interop.
// Color is specified as NV12 luma/chroma (Y,U,V).
// - eglImage: EGLImageKHR from nvivafilter callback
// - W,H: frame size
// - arrows: CPU array
// - n: number of arrows
// - Y,U,V: desired arrow color in YUV (NV12)
void overlay_draw_arrows_nv12(EGLImageKHR eglImage,
                              int W, int H,
                              const Arrow *arrows, int n,
                              uint8_t Y, uint8_t U, uint8_t V);

#ifdef __cplusplus
}
#endif
