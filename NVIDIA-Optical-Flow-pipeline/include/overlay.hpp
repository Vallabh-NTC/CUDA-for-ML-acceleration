#pragma once
#include <cstdint>

// NVCC spesso non vede EGLImageKHR se non includi EGL headers.
// Per evitare include pesanti in .cu, facciamo forward typedef.
typedef void* EGLImageKHR;

struct Arrow
{
    int x0, y0, x1, y1;
};

#ifdef __cplusplus
extern "C" {
#endif

// Draws arrows on NV12 luma (Y) plane in-place on the EGLImage.
// - eglImage: EGLImageKHR from nvivafilter callback
// - W,H: frame size
// - arrows: CPU array
// - n: number of arrows
// - colorY: luma value (0..255)
void overlay_draw_arrows_nv12(EGLImageKHR eglImage,
                              int W, int H,
                              const Arrow *arrows, int n,
                              uint8_t colorY);

#ifdef __cplusplus
}
#endif
