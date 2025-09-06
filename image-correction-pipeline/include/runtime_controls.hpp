// image-correction-pipeline/include/runtime_controls.hpp
#pragma once
#include <string>
#include <atomic>
#include <thread>
#include <filesystem>
#include "rectify_config.hpp"

// ----------------------------------------------------------------------------
// RuntimeControls
// ----------------------------------------------------------------------------
// Purpose:
//   - Own a RectifyConfig that is *hot-reloaded* from a JSON file on disk.
//   - A lightweight background thread polls the file's modification time and
//     reloads the values when changed. Only the keys we care about are parsed.
// 
// Thread-safety model:
//   - The config struct is a small POD; we update it by assignment, and readers
//     (the GPU process callback) take a by-value snapshot via get().
//   - Polling interval is short (200 ms), but only stat() happens each cycle;
//     actual file read/parse only occurs when mtime changes.
// ----------------------------------------------------------------------------
class RuntimeControls {
public:
    // 'path' is the absolute path to your controls.json (e.g., /opt/rectify/controls.json)
    explicit RuntimeControls(const std::string& path);
    ~RuntimeControls();

    // Returns the latest configuration snapshot. Copy is cheap and safe.
    RectifyConfig get() const { return cfg_; }

private:
    void watch_loop();

    // Minimal tolerant loader for flat float keys. Missing keys are ignored.
    static bool load_controls_from_file(const std::string& path, RectifyConfig& rc);

    std::string path_;                  // JSON file to watch
    std::atomic<bool> running_{false};  // thread control flag
    std::thread watcher_;               // background polling thread
    mutable RectifyConfig cfg_{};       // current config (read mostly, updated atomically)
};
