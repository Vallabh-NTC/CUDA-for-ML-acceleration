// src/main.cu
#include <gst/gst.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "gstreamer_pipeline.hpp"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::cerr << "CUDA error: " << cudaGetErrorString(e)                  \
            << " at " << __FILE__ << ":" << __LINE__ << "\n";           \
  std::exit(1);} } while(0)

// Add `add_val` to RGB channels of RGBA8 image (clamped). Alpha unchanged.
__global__ void brightenRGBA(uint8_t* rgba, int num_pixels, int add_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;
    int base = idx * 4;
    unsigned int r = rgba[base + 0];
    unsigned int g = rgba[base + 1];
    unsigned int b = rgba[base + 2];
    r = min(r + (unsigned int)add_val, 255u);
    g = min(g + (unsigned int)add_val, 255u);
    b = min(b + (unsigned int)add_val, 255u);
    rgba[base + 0] = static_cast<uint8_t>(r);
    rgba[base + 1] = static_cast<uint8_t>(g);
    rgba[base + 2] = static_cast<uint8_t>(b);
}

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    // Enable host-mapped (zero-copy) memory before any CUDA allocs
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    // RGBA to appsink in system memory (no NVMM on output caps)
    const char* pipeline_desc =
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=RGBA ! "
        "appsink name=mysink sync=false max-buffers=1 drop=true";

    GStreamerCamera cam(pipeline_desc);
    if (!cam.start()) {
        std::cerr << "Failed to start camera pipeline\n";
        return 1;
    }

    // Grab one frame into pinned host memory (and get device alias)
    uint8_t* d_ptr = nullptr;   // device alias to pinned memory (zero-copy)
    uint8_t* h_ptr = nullptr;   // host pointer to pinned memory
    size_t   nbytes = 0;
    int w = 0, h = 0;

    if (!cam.grab_frame_to_pinned(&d_ptr, &h_ptr, &nbytes, &w, &h)) {
        std::cerr << "Failed to grab frame\n";
        cam.stop();
        return 1;
    }

    const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h) * 4;
    if (nbytes < expected) {
        std::cerr << "Unexpected buffer size: got " << nbytes
                  << ", expected >= " << expected << "\n";
        cam.stop();
        return 1;
    }

    // Launch kernel that writes directly into pinned memory via d_ptr
    int num_pixels = w * h;
    int threads = 256;
    int blocks  = (num_pixels + threads - 1) / threads;
    brightenRGBA<<<blocks, threads>>>(d_ptr, num_pixels, /*add_val=*/125);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // ensure GPU writes are visible on CPU

    // Write PPM (P6) from h_ptr directly (skip any cudaMemcpy)
    std::ofstream ofs("frame_after.ppm", std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open output file\n";
        cam.stop();
        return 1;
    }
    ofs << "P6\n" << w << " " << h << "\n255\n";

    // pack-and-write one row at a time (RGB only), avoiding a full extra image copy
    std::vector<uint8_t> row(3 * w);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = h_ptr + static_cast<size_t>(y) * w * 4;
        for (int x = 0; x < w; ++x) {
            const uint8_t* px = &src[x * 4];
            row[x * 3 + 0] = px[0]; // R
            row[x * 3 + 1] = px[1]; // G
            row[x * 3 + 2] = px[2]; // B
        }
        ofs.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
    ofs.close();

    cam.stop();
    std::cout << "Wrote frame_after.ppm (" << w << "x" << h << ")\n";
    return 0;
}
