#include <gst/gst.h>
#include <iostream>
#include <cstdio>              // for device-side printf formatting
#include <cuda_runtime.h>      // CUDA runtime
#include "gstreamer_pipeline.hpp"

// --- helpers ---
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) {                                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e)                 \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

// Kernel: print thread/block IDs and array contents (if in range)
__global__ void debugPrintKernel(const int* arr, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        printf("[block %d/%d] [thread %d/%d] gid=%d  arr[%d]=%d\n",
               bid, gridDim.x, tid, blockDim.x, gid, tid, arr[tid]);
    } else {
        printf("[block %d/%d] [thread %d/%d] gid=%d  arr[%d]=<out-of-range>\n",
               bid, gridDim.x, tid, blockDim.x, gid, tid);
    }
}

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    // --- CUDA zero-copy setup on Xavier ---
    // Must be called before any context-creating CUDA call.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    // 1) Allocate pinned, host-mapped memory
    const int N = 10;
    int* h_arr = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_arr),
                             N * sizeof(int),
                             cudaHostAllocMapped)); // pinned + mapped

    // Initialize host data
    for (int i = 0; i < N; ++i) h_arr[i] = i; // 0..9

    // 2) Get the device alias for the same host memory
    int* d_alias = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_alias),
                                        h_arr, 0));

    // 3) Launch 1 block, 15 threads, and print from inside the kernel
    dim3 grid(1), block(15);
    debugPrintKernel<<<grid, block>>>(d_alias, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // flush device-side printf

    // Clean up pinned memory (after kernel is done)
    CUDA_CHECK(cudaFreeHost(h_arr));

    // ---- Your original GStreamer pipeline (unchanged behavior) ----
    const char* pipeline_desc =
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=I420 ! "
        "xvimagesink sync=false";

    GStreamerCamera cam(pipeline_desc);
    bool ok = cam.run();

    return ok ? 0 : 1;
}
