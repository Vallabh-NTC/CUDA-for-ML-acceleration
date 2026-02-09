// overlay.hpp
#pragma once
#include <cstdint>

// Forward typedef to avoid pulling EGL headers into CUDA TU headers.
typedef void* EGLImageKHR;

#ifdef __cplusplus
extern "C" {
#endif

// Draw MV field arrows and one OF resultant arrow + optional IMU resultant arrow.
// mvPtr points to device memory: 2S16 interleaved (dx,dy) in S10.5.
// mvPitchBytes is pitch in bytes of mvPtr.
//
// NOTE:
// - OF resultant vector is in px/frame.
// - IMU resultant vector (imuDx, imuDy) is in m/s^2 (linear accel) filtered,
//   so it's purely visual. Use imuScale to convert into pixels.
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
                           // IMU resultant (optional): pass 0,0 to disable
                           float imuDx, float imuDy,
                           float imuScale,
                           uint8_t imuY, uint8_t imuU, uint8_t imuV,
                           // Visual-only direction forcing options
                           float forceDeg,
                           float forceMinResMag,
                           uint32_t frameTag);

#ifdef __cplusplus
}
#endif
