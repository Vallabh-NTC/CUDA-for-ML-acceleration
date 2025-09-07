#pragma once
// Minimal ABI mirror for gst-nvivafilter customer library (JetPack 5.x)
// We avoid depending on NVIDIA headers that are no longer shipped with samples.

#include <cstdint>

// EGLImageKHR is an opaque handle; forward-declare to avoid EGL headers dependency.
typedef void* EGLImageKHR;

// ColorFormat is passed by pointer but we don't dereference it here;
// keep a minimal enum to satisfy the ABI.
enum ColorFormat : uint32_t {
    COLOR_FORMAT_UNKNOWN = 0,
    COLOR_FORMAT_ABGR    = 1,  // matches CU_EGL_COLOR_FORMAT_ABGR expectation in our code
};

// Function pointer types expected by gst-nvivafilter
using PreProcessFunc = void(*)(void ** /*inOutBufs*/,
                               unsigned int* /*width*/,
                               unsigned int* /*height*/,
                               unsigned int* /*pitch*/,
                               unsigned int* /*size*/,
                               ColorFormat*  /*colorFormat*/,
                               unsigned int  /*batchSize*/,
                               void **       /*usrptr*/);

using GPUProcessFunc = void(*)(EGLImageKHR /*image*/, void ** /*usrptr*/);

using PostProcessFunc = void(*)(void ** /*inOutBufs*/,
                                unsigned int* /*width*/,
                                unsigned int* /*height*/,
                                unsigned int* /*pitch*/,
                                unsigned int* /*size*/,
                                ColorFormat*  /*colorFormat*/,
                                unsigned int  /*batchSize*/,
                                void **       /*usrptr*/);

// Struct filled by our init(); gst-nvivafilter will call our callbacks.
struct CustomerFunction {
    PreProcessFunc  fPreProcess  = nullptr;
    GPUProcessFunc  fGPUProcess  = nullptr;
    PostProcessFunc fPostProcess = nullptr;
};

// The .so must export these two symbols with C linkage.
extern "C" void init(CustomerFunction* f);
extern "C" void deinit(void);
