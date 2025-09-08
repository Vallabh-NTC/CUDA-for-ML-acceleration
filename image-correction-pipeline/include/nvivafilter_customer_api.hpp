/**
 * @file nvivafilter_customer_api.hpp
 * @brief Official NVIDIA "nvivafilter" customer API stub.
 *
 * GStreamer’s `nvivafilter` plugin loads this library and expects a standard C API:
 *   - init(CustomerFunction*)
 *   - deinit()
 *
 * The CustomerFunction struct contains function pointers:
 *   - fPreProcess   → called before GPU mapping
 *   - fGPUProcess   → called with mapped CUDA/EGL frame
 *   - fPostProcess  → called after GPU processing
 *
 * Our implementation (`nvivafilter_imagecorrection.cpp`) fills these hooks with
 * our CUDA pipeline. This header is provided by NVIDIA’s Multimedia API and is
 * included here for completeness.
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
  /**
  * cuda-process API
  *
  * @param image   : EGL Image to process
  * @param userPtr : point to user alloc data, should be free by user
  */
  void (*fGPUProcess) (EGLImageKHR image, void ** userPtr);

  /**
  * pre-process API
  *
  * @param sBaseAddr  : Mapped Surfaces(YUV) pointers
  * @param smemsize   : surfaces size array
  * @param swidth     : surfaces width array
  * @param sheight    : surfaces height array
  * @param spitch     : surfaces pitch array
  * @param sformat    : surfaces format array
  * @param nsurfcount : surfaces count
  * @param userPtr    : point to user alloc data, should be free by user
  */
  void (*fPreProcess)(void **sBaseAddr,
                      unsigned int *smemsize,
                      unsigned int *swidth,
                      unsigned int *sheight,
                      unsigned int *spitch,
                      ColorFormat *sformat,
                      unsigned int nsurfcount,
                      void ** userPtr);

  /**
  * post-process API
  *
  * @param sBaseAddr  : Mapped Surfaces(YUV) pointers
  * @param smemsize   : surfaces size array
  * @param swidth     : surfaces width array
  * @param sheight    : surfaces height array
  * @param spitch     : surfaces pitch array
  * @param sformat    : surfaces format array
  * @param nsurfcount : surfaces count
  * @param userPtr    : point to user alloc data, should be free by user
  */
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
