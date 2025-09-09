/**
 * @file nvivafilter_customer_api.hpp
 * @brief NVIDIA "nvivafilter" customer API definition.
 *
 * GStreamer’s `nvivafilter` plugin loads this library and expects:
 *   - init(CustomerFunction*)
 *   - deinit()
 *
 * The CustomerFunction struct contains function pointers:
 *   - fPreProcess   → called before GPU mapping
 *   - fGPUProcess   → called with CUDA/EGL mapped frame
 *   - fPostProcess  → called after GPU processing
 *
 * Our implementation (`nvivafilter_imagecorrection.cpp`) fills these hooks
 * with our CUDA rectification + enhancement pipeline.
 */

#ifndef _CUSTOMER_FUNCTIONS_H_
#define _CUSTOMER_FUNCTIONS_H_

#include <cudaEGL.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
  COLOR_FORMAT_Y8 = 0,
  COLOR_FORMAT_U8_V8,
  COLOR_FORMAT_RGBA,
  COLOR_FORMAT_NONE
} ColorFormat;

typedef struct {
  void (*fGPUProcess) (EGLImageKHR image, void ** userPtr);
  void (*fPreProcess)(void **sBaseAddr,
                      unsigned int *smemsize,
                      unsigned int *swidth,
                      unsigned int *sheight,
                      unsigned int *spitch,
                      ColorFormat *sformat,
                      unsigned int nsurfcount,
                      void ** userPtr);
  void (*fPostProcess)(void **sBaseAddr,
                      unsigned int *smemsize,
                      unsigned int *swidth,
                      unsigned int *sheight,
                      unsigned int *spitch,
                      ColorFormat *sformat,
                      unsigned int nsurfcount,
                      void ** userPtr);
} CustomerFunction;

void init (CustomerFunction * pFuncs);
void deinit (void);

#if defined(__cplusplus)
}
#endif

#endif
