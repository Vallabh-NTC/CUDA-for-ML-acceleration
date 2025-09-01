// src/main.cu
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
#include <cmath>
#include <cstdio>   // for std::rename (atomic replace on POSIX)

#include "gstreamer_pipeline.hpp"
#include "image_correction.hpp"
#include "runtime_controls.hpp"  // runtime color controls + file watcher

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::cerr << "CUDA error: " << cudaGetErrorString(e)                  \
            << " at " << __FILE__ << ":" << __LINE__ << "\n";           \
  std::exit(1);} } while(0)

using icp::RectifyConfig;
using icp::fisheye_rectify_rgba;

// -------- pipeline builder (RGBA appsinks) --------
static std::string build_multi_pipeline(int w, int h, int num, int den, int cams) {
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

// -------- simple PPM writer (RGB from RGBA) --------
static void write_ppm(const std::string& path, const uint8_t* rgba, int w, int h) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) { std::cerr << "open " << path << " failed\n"; return; }
    ofs << "P6\n" << w << " " << h << "\n255\n";
    std::vector<uint8_t> row(3 * (size_t)w);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgba + (size_t)y * w * 4;
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = &src[x * 4];
            row[x*3+0] = p[0]; row[x*3+1] = p[1]; row[x*3+2] = p[2];
        }
        ofs.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
}

// Atomic replace: write to temporary file then rename to final path
static void write_ppm_atomic(const std::string& final_path,
                             const uint8_t* rgba, int w, int h) {
    std::string tmp = final_path + ".tmp";
    write_ppm(tmp, rgba, w, h);
    std::rename(tmp.c_str(), final_path.c_str());
}

struct CameraJob {
    int idx = 0;
    int frames_to_process = -1; // -1 = run forever
};

static void cam_worker(GStreamerMultiCamera* multi,
                       CameraJob job,
                       const RectifyConfig& cfg,
                       RuntimeControls* runtime_ctrls,
                       std::atomic<bool>& running)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Output pinned (host-mapped) buffer for rectified frames
    uint8_t* h_out = nullptr;   // host pointer
    uint8_t* d_out = nullptr;   // device alias
    size_t   out_capacity = 0;  // bytes

    auto ensure_out_buffer = [&](int dst_w, int dst_h) {
        const size_t need = (size_t)dst_w * dst_h * 4;
        if (need <= out_capacity) return true;
        if (h_out) { cudaFreeHost(h_out); h_out = nullptr; d_out = nullptr; out_capacity = 0; }
        if (cudaHostAlloc((void**)&h_out, need, cudaHostAllocMapped) != cudaSuccess) return false;
        void* alias = nullptr;
        if (cudaHostGetDevicePointer(&alias, h_out, 0) != cudaSuccess) return false;
        d_out = static_cast<uint8_t*>(alias);
        out_capacity = need;
        return true;
    };

    // --- throttled save config: one single file per camera ---
    constexpr int SAVE_INTERVAL_MS = 50; // write at most every 50 ms
    auto last_save = std::chrono::steady_clock::now() - std::chrono::milliseconds(SAVE_INTERVAL_MS);
    const std::string out_filename = "cam" + std::to_string(job.idx) + "_latest.ppm";

    int frame_idx = 0;
    while (running.load(std::memory_order_relaxed)) {
        if (job.frames_to_process >= 0 && frame_idx >= job.frames_to_process) break;

        uint8_t *d_src=nullptr, *h_src=nullptr;
        size_t nbytes=0; int w=0, h=0;

        // Grab one RGBA frame into pinned memory (d_src is device alias)
        if (!multi->grab_frame_to_pinned(job.idx, &d_src, &h_src, &nbytes, &w, &h)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        const size_t expected = (size_t)w * h * 4;
        if (nbytes < expected) continue;

        // Destination dims: fixed width (1920) with input aspect ratio
        const int dst_w = cfg.out_width;
        const int dst_h = (int)std::lround(dst_w * (double)h / (double)w);
        if (!ensure_out_buffer(dst_w, dst_h)) {
            std::cerr << "[cam " << job.idx << "] out buffer alloc failed\n";
            break;
        }

        const size_t src_stride = (size_t)w     * 4;
        const size_t dst_stride = (size_t)dst_w * 4;
        
        // Take a per-frame snapshot of runtime color controls
        float bri, con, sat, gam, wbr, wbg, wbb;
        runtime_ctrls->snapshot(bri, con, sat, gam, wbr, wbg, wbb);

        // Make a local copy of cfg and override color fields
        RectifyConfig local_cfg = cfg;
        local_cfg.brightness = bri;
        local_cfg.contrast   = con;
        local_cfg.saturation = sat;
        local_cfg.gamma      = gam;
        local_cfg.wb_r = wbr; local_cfg.wb_g = wbg; local_cfg.wb_b = wbb;

        // Undistort (equidistant -> perspective) to d_out
        fisheye_rectify_rgba(
            d_src,  w,     h,     src_stride,
            d_out,  dst_w, dst_h, dst_stride,
            local_cfg,
            stream
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream)); // make results visible in h_out

        // Throttled single-file save
        auto now = std::chrono::steady_clock::now();
        if (now - last_save >= std::chrono::milliseconds(SAVE_INTERVAL_MS)) {
            write_ppm_atomic(out_filename, h_out, dst_w, dst_h);
            last_save = now;
        }

        ++frame_idx;
    }

    if (h_out) cudaFreeHost(h_out);
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "[cam " << job.idx << "] worker done\n";
}

int main(int argc, char** argv) {
    gst_init(&argc, &argv);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    // Runtime color controls (hot-adjustable, shared by all workers)
    RuntimeControls runtime_ctrls;

    // Optional: initialize from a controls.json if present
    load_controls_from_file("controls.json", runtime_ctrls);

    std::atomic<bool> running{true};

    // Start a background watcher that reloads controls.json on change
    std::thread ctrl_watcher([&]{
        watch_controls_file("controls.json", runtime_ctrls, running);
    });

    // Source camera setup
    const int SRC_W = 1920;
    const int SRC_H = 1080;
    const int FPS_NUM = 30;
    const int FPS_DEN = 1;
    const int NUM_CAMS = 1;

    // Integrated rectification parameters (fixed)
    RectifyConfig rectify_cfg{};
    rectify_cfg.fish_fov_deg = 195.1f;
    rectify_cfg.out_hfov_deg = 90.0f;
    rectify_cfg.cx_f = 959.50f;
    rectify_cfg.cy_f = 539.50f;
    rectify_cfg.r_f  = 1100.77f;
    rectify_cfg.out_width = 1920;

    // --- Default color controls (can be overridden by controls.json) ---
    rectify_cfg.brightness = 0.0f;
    rectify_cfg.contrast   = 1.0f;
    rectify_cfg.saturation = 1.0f;
    rectify_cfg.gamma      = 1.0f;
    rectify_cfg.wb_r = 1.0f;
    rectify_cfg.wb_g = 1.0f;
    rectify_cfg.wb_b = 1.0f;

    // Build multi-camera pipeline and sinks
    std::string pipeline = build_multi_pipeline(SRC_W, SRC_H, FPS_NUM, FPS_DEN, NUM_CAMS);
    std::vector<std::string> sinks;
    sinks.reserve(NUM_CAMS);
    for (int i = 0; i < NUM_CAMS; ++i) sinks.emplace_back("mysink" + std::to_string(i));

    GStreamerMultiCamera multi(pipeline, sinks);
    if (!multi.start()) {
        std::cerr << "Failed to start multi-camera pipeline\n";
        return 1;
    }

    // Frames to process (optional): argv[1], default = run forever
    int frames = -1;
    if (argc >= 2) { try { frames = std::stoi(argv[1]); } catch (...) {} }

    std::vector<std::thread> threads;
    threads.reserve(NUM_CAMS);
    for (int i = 0; i < NUM_CAMS; ++i) {
        threads.emplace_back(cam_worker,
                             &multi,
                             CameraJob{ .idx = i, .frames_to_process = frames },
                             std::cref(rectify_cfg),
                             &runtime_ctrls,
                             std::ref(running));
    }
    for (auto& t : threads) t.join();
    
    running.store(false, std::memory_order_relaxed);
    if (ctrl_watcher.joinable()) ctrl_watcher.join();

    multi.stop();
    std::cout << "All done.\n";
    return 0;
}
