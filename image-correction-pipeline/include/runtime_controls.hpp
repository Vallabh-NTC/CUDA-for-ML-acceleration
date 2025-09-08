/**
 * @file runtime_controls.hpp
 * @brief Hot-reload JSON configuration for nvivafilter_imagecorrection.
 *
 * Overview:
 * ---------
 *  This class enables live configuration of the CUDA image correction
 *  pipeline (rectification + color grading + auto exposure).
 *
 *  - A JSON file on disk defines the parameters (see RectifyConfig in
 *    rectify_config.hpp).
 *  - The file path is passed via environment variable `ICP_CONTROLS`.
 *  - At startup, RuntimeControls loads the JSON once.
 *  - If "watch" is enabled, a background thread monitors the file’s
 *    modification time. When the file changes, the JSON is reloaded
 *    and the new values replace the current configuration.
 *
 * Why:
 * ----
 *  This avoids recompiling or restarting GStreamer when fine-tuning
 *  exposure, contrast, white balance, fisheye calibration, etc.
 *
 * Typical usage in nvivafilter_imagecorrection.cpp:
 * -------------------------------------------------
 *  if (const char* p = getenv("ICP_CONTROLS")) {
 *      g_rc = new RuntimeControls(std::string(p), true);
 *  }
 *  ...
 *  icp::RectifyConfig cfg = g_rc ? g_rc->get() : g_cfg;
 *
 * Thread safety:
 * --------------
 *  - cfg_ is updated atomically when JSON is reloaded.
 *  - get() returns the latest snapshot.
 *
 * Dependencies:
 * -------------
 *  - C++17 filesystem for modification time.
 *  - nlohmann/json (or custom parser) for JSON decode.
 */
 
#pragma once
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <filesystem>  // <-- C++17
#include "rectify_config.hpp"   // icp::RectifyConfig

class RuntimeControls {
public:
    explicit RuntimeControls(std::string path, bool enable_watch = false);
    ~RuntimeControls();

    icp::RectifyConfig get() const;
    bool reload();

    static bool load_controls_from_file(const std::string& path, icp::RectifyConfig& rc);

private:
    void watch_loop();

    std::string path_;
    mutable std::mutex mtx_;
    icp::RectifyConfig cfg_{};   // stato corrente

    std::atomic<bool> stop_{false};
    std::thread th_;

    // In C++17 usiamo direttamente il tipo di filesystem
    std::filesystem::file_time_type last_mtime_{};
};
