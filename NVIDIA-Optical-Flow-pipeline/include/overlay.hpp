// overlay.hpp
#pragma once
#include <cstdint>

// Forward typedef to avoid pulling EGL headers into CUDA TU headers.
typedef void* EGLImageKHR;

#ifdef __cplusplus
extern "C" {
#endif

// Draw MV field arrows and one resultant arrow directly from pitch-linear MV buffer.
// mvPtr points to device memory: 2S16 interleaved (dx,dy) in S10.5.
// mvPitchBytes is pitch in bytes of mvPtr.
//
void overlay_draw_mvs_nv12(EGLImageKHR eglImage,
                           int W, int H,
                           const int16_t *mvPtr, int mvPitchBytes,
                           int mvW, int mvH, int grid,
                           int x0, int x1, int y0, int y1,
                           int step, float scale,
                           float minMagDraw,
                           uint8_t fieldY, uint8_t fieldU, uint8_t fieldV,
                           float resDxPx, float resDyPx,
                           uint8_t resY, uint8_t resU, uint8_t resV,
                           float forceDeg,
                           float forceMinResMag,
                           uint32_t frameTag);

#ifdef __cplusplus
}
#endif
