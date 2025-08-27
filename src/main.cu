#include <gst/gst.h>
#include <cuda_runtime.h>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include "gstreamer_pipeline.hpp"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::cerr << "CUDA error: " << cudaGetErrorString(e)                  \
            << " at " << __FILE__ << ":" << __LINE__ << "\n";           \
  std::exit(1);} } while(0)

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

struct CameraCfg {
    int idx;        // 0..N-1
    int brighten{125};
    int frames_to_process{200}; // -1 for run forever
};

static std::string build_multi_pipeline(int w, int h, int num, int den, int cams) {
    // Parallel branches in one description. Each branch ends in RGBA appsink.
    // NVMM on source, system memory at appsink.
    std::ostringstream ss;
    for (int i = 0; i < cams; ++i) {
        ss << "nvarguscamerasrc sensor-id=" << i << " ! "
           << "video/x-raw(memory:NVMM),width=" << w << ",height=" << h
           << ",framerate=" << num << "/" << den << " ! "
           << "nvvidconv ! "
           << "video/x-raw,format=RGBA ! "
           << "appsink name=mysink" << i
           << " sync=false max-buffers=1 drop=true ";
    }
    return ss.str();
}

static void write_ppm(const std::string& path, const uint8_t* rgba, int w, int h) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) { std::cerr << "open " << path << " failed\n"; return; }
    ofs << "P6\n" << w << " " << h << "\n255\n";
    std::vector<uint8_t> row(3 * w);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgba + static_cast<size_t>(y) * w * 4;
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = &src[x * 4];
            row[x*3+0] = p[0]; row[x*3+1] = p[1]; row[x*3+2] = p[2];
        }
        ofs.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
}

static void cam_worker(GStreamerMultiCamera* multi,
                       CameraCfg cfg,
                       std::atomic<bool>& running)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    int frame_idx = 0;
    while (running.load(std::memory_order_relaxed)) {
        if (cfg.frames_to_process >= 0 && frame_idx >= cfg.frames_to_process) break;

        uint8_t *d_ptr=nullptr, *h_ptr=nullptr;
        size_t nbytes=0; int w=0, h=0;

        if (!multi->grab_frame_to_pinned(cfg.idx, &d_ptr, &h_ptr, &nbytes, &w, &h)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        const size_t expected = static_cast<size_t>(w) * h * 4;
        if (nbytes < expected) continue;

        int num_pixels = w * h;
        int threads = 256;
        int blocks  = (num_pixels + threads - 1) / threads;
        brightenRGBA<<<blocks, threads, 0, stream>>>(d_ptr, num_pixels, cfg.brighten);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream)); // make writes visible to CPU

        std::ostringstream name;
        name << "cam" << cfg.idx << "_frame" << std::setw(6) << std::setfill('0') << frame_idx << ".ppm";
        write_ppm(name.str(), h_ptr, w, h);
        ++frame_idx;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "[cam " << cfg.idx << "] worker done\n";
}

int main(int argc, char** argv) {
    gst_init(&argc, &argv);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    const int W=1920, H=1080, NUM=30, DEN=1, CAMS=3;

    // Build one pipeline containing all branches
    std::string pipe = build_multi_pipeline(W, H, NUM, DEN, CAMS);

    // appsink names in same order
    std::vector<std::string> sinks;
    for (int i=0; i<CAMS; ++i) sinks.emplace_back("mysink"+std::to_string(i));

    GStreamerMultiCamera multi(pipe, sinks);
    if (!multi.start()) {
        std::cerr << "Failed to start multi-camera pipeline\n";
        return 1;
    }

    std::atomic<bool> running{true};
    int frames = (argc >= 2) ? std::stoi(argv[1]) : 200;

    std::vector<std::thread> threads;
    for (int i=0;i<CAMS;++i) {
        threads.emplace_back(cam_worker, &multi, CameraCfg{.idx=i, .brighten=125, .frames_to_process=frames}, std::ref(running));
    }
    for (auto& t : threads) t.join();

    multi.stop();
    std::cout << "All done.\n";
    return 0;
}
