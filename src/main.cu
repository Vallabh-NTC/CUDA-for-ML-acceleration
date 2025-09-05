// src/main.cu
#include <gst/gst.h>
#include <cuda_runtime.h>

#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <memory> // std::unique_ptr

#include "gstreamer_pipeline.hpp"  // appsink multi-camera helper
#include "image_correction.hpp"    // fisheye_rectify_rgba + RectifyConfig
#include "runtime_controls.hpp"    // live color controls + file watcher
#include "rtp_sender.hpp"          // RTP sender (H.264 only)

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::cerr << "CUDA error: " << cudaGetErrorString(e)                  \
            << " at " << __FILE__ << ":" << __LINE__ << "\n";           \
  std::exit(1);} } while(0)

using icp::RectifyConfig;
using icp::fisheye_rectify_rgba;

// -----------------------------------------------------------------------------
// Build a multi-camera pipeline (each branch ends with a unique RGBA appsink).
// Uses nvvidconv (name present on your Jetson).
// -----------------------------------------------------------------------------
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

struct CameraJob {
    int idx = 0;
    int frames_to_process = -1; // -1 = run forever
};

// -----------------------------------------------------------------------------
// Worker:
// 1) Pull RGBA from appsink into pinned host memory (device alias available)
// 2) Rectify (CUDA) into another pinned host-mapped buffer (h_out/d_out)
// 3) Push the rectified frame to RTP sender (H.264) over UDP
// -----------------------------------------------------------------------------
static void cam_worker(GStreamerMultiCamera* multi,
                       CameraJob job,
                       const RectifyConfig& cfg_base,
                       RuntimeControls* runtime_ctrls,
                       std::atomic<bool>& running,
                       const std::string& dst_ip,
                       int dst_port,
                       int fps_num, int fps_den)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Output pinned (host-mapped) buffer for rectified frames
    uint8_t* h_out = nullptr;   // host pointer
    uint8_t* d_out = nullptr;   // device alias (via cudaHostGetDevicePointer)
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

    // RTP sender is created lazily on first frame (when output size is known)
    std::unique_ptr<RtpSender> sender;

    int frame_idx = 0;
    while (running.load(std::memory_order_relaxed)) {
        if (job.frames_to_process >= 0 && frame_idx >= job.frames_to_process) break;

        uint8_t *d_src=nullptr, *h_src=nullptr;
        size_t nbytes=0; int w=0, h=0;

        // Pull one RGBA frame into pinned host; d_src is device alias of the same memory
        if (!multi->grab_frame_to_pinned(job.idx, &d_src, &h_src, &nbytes, &w, &h)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        const size_t expected = (size_t)w * h * 4;
        if (nbytes < expected) continue; // corrupted or partial buffer

        // Destination dimensions: fixed width, preserve input aspect ratio
        const int dst_w = cfg_base.out_width;
        const int dst_h = static_cast<int>(std::lround(dst_w * (double)h / (double)w));
        if (!ensure_out_buffer(dst_w, dst_h)) {
            std::cerr << "[cam " << job.idx << "] output buffer allocation failed\n";
            break;
        }

        const size_t src_stride = (size_t)w     * 4;
        const size_t dst_stride = (size_t)dst_w * 4;

        // Snapshot runtime color controls (atomic -> plain)
        float bri, con, sat, gam, wbr, wbg, wbb;
        runtime_ctrls->snapshot(bri, con, sat, gam, wbr, wbg, wbb);

        // Copy base cfg and override color fields per frame
        RectifyConfig local_cfg = cfg_base;
        local_cfg.brightness = bri;
        local_cfg.contrast   = con;
        local_cfg.saturation = sat;
        local_cfg.gamma      = gam;
        local_cfg.wb_r = wbr; local_cfg.wb_g = wbg; local_cfg.wb_b = wbb;

        // Rectify (equidistant -> perspective) into d_out (visible on host as h_out)
        fisheye_rectify_rgba(
            d_src,  w,     h,     src_stride,
            d_out,  dst_w, dst_h, dst_stride,
            local_cfg,
            stream
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream)); // ensure h_out is filled

        // Create RTP sender on first frame when output size is known
        if (!sender) {
            sender = std::make_unique<RtpSender>(dst_ip, dst_port, dst_w, dst_h, fps_num, fps_den);
            if (!sender->start()) {
                std::cerr << "[cam " << job.idx << "] RTP sender start failed\n";
                break;
            }
            std::cout << "[cam " << job.idx << "] streaming rectified "
                      << dst_w << "x" << dst_h << "@" << fps_num << "/" << fps_den
                      << " via H.264 to " << dst_ip << ":" << dst_port << "\n";
        }

        // Push the rectified image to RTP pipeline
        if (!sender->push_rgba(h_out, dst_w, dst_h, (int)dst_stride)) {
            std::cerr << "[cam " << job.idx << "] RTP push failed\n";
        }

        ++frame_idx;
    }

    if (sender) sender->stop();
    if (h_out) cudaFreeHost(h_out);
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "[cam " << job.idx << "] worker done\n";
}

int main(int argc, char** argv) {
    // -------- initialization --------
    gst_init(&argc, &argv);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    // Live color controls shared by all workers
    RuntimeControls runtime_ctrls;
    // Optional: initialize from "controls.json" in current working directory
    load_controls_from_file("controls.json", runtime_ctrls);

    std::atomic<bool> running{true};

    // Background watcher to hot-reload "controls.json" on change
    std::thread ctrl_watcher([&]{
        watch_controls_file("controls.json", runtime_ctrls, running);
    });

    // -------- source camera configuration --------
    const int SRC_W = 1920;
    const int SRC_H = 1080;
    const int FPS_NUM = 30;
    const int FPS_DEN = 1;

    // Number of cameras you want to run (1..6). Each cam maps to UDP port 5000+i.
    const int NUM_CAMS = 3;   // <-- set this to 1..6 as needed

    // Sanity: limit to supported RTP port range (5000..5005)
    const int RTP_BASE_PORT = 5000;
    const int RTP_MAX_CAMS  = 6; // cams 0..5
    if (NUM_CAMS > RTP_MAX_CAMS) {
        std::cerr << "[error] NUM_CAMS=" << NUM_CAMS
                  << " exceeds supported RTP port range ("
                  << RTP_MAX_CAMS << " cams -> ports 5000..5005).\n";
        running.store(false, std::memory_order_relaxed);
        if (ctrl_watcher.joinable()) ctrl_watcher.join();
        return 1;
    }

    // -------- rectification parameters (fixed optics) --------
    RectifyConfig rectify_cfg{};
    rectify_cfg.fish_fov_deg = 195.1f;
    rectify_cfg.out_hfov_deg = 90.0f;
    rectify_cfg.cx_f = 959.50f;
    rectify_cfg.cy_f = 539.50f;
    rectify_cfg.r_f  = 1100.77f;
    rectify_cfg.out_width = 1920;
    

    // Default color controls (can be overridden at runtime by controls.json)
    rectify_cfg.brightness = 0.0f;
    rectify_cfg.contrast   = 1.0f;
    rectify_cfg.saturation = 1.0f;
    rectify_cfg.gamma      = 1.0f;
    rectify_cfg.wb_r = 1.0f;
    rectify_cfg.wb_g = 1.0f;
    rectify_cfg.wb_b = 1.0f;
    

    // -------- build GStreamer pipeline with N branches (one per camera) --------
    (void)SRC_H; // avoid unused warning
    std::string pipeline = build_multi_pipeline(SRC_W, SRC_H, FPS_NUM, FPS_DEN, NUM_CAMS);
    std::vector<std::string> sinks;
    sinks.reserve(NUM_CAMS);
    for (int i = 0; i < NUM_CAMS; ++i) sinks.emplace_back("mysink" + std::to_string(i));

    GStreamerMultiCamera multi(pipeline, sinks);
    if (!multi.start()) {
        std::cerr << "Failed to start multi-camera pipeline\n";
        running.store(false, std::memory_order_relaxed);
        if (ctrl_watcher.joinable()) ctrl_watcher.join();
        return 1;
    }

    // Optional: limit number of frames via argv[1]; default: run forever
    int frames = -1;
    if (argc >= 2) { try { frames = std::stoi(argv[1]); } catch (...) {} }

    // -------- RTP destination config --------
    const std::string dst_ip   = "192.168.10.201"; // <-- set to your Windows PC IP

    // -------- launch camera workers --------
    std::vector<std::thread> threads;
    threads.reserve(NUM_CAMS);
    for (int i = 0; i < NUM_CAMS; ++i) {
        const int port = RTP_BASE_PORT + i;  // cam i -> port 5000+i
        std::cout << "[info] cam" << i << " -> rtp://" << dst_ip << ":" << port << "\n";

        CameraJob job;
        job.idx = i;
        job.frames_to_process = frames;

        threads.emplace_back(cam_worker,
                             &multi,
                             job,
                             std::cref(rectify_cfg),
                             &runtime_ctrls,
                             std::ref(running),
                             dst_ip,
                             port,
                             FPS_NUM, FPS_DEN);
    }

    // Wait for all workers
    for (auto& t : threads) t.join();

    // -------- teardown --------
    running.store(false, std::memory_order_relaxed);
    if (ctrl_watcher.joinable()) ctrl_watcher.join();

    multi.stop();
    std::cout << "All done.\n";
    return 0;
}
