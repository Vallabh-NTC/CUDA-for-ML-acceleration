#pragma once
#include <string>
#include <chrono>
#include "color_ops.cuh"

namespace icp {

// Minimal file watcher + flat JSON reader. No external dependencies.
class RuntimeControls {
public:
    explicit RuntimeControls(std::string path = "/home/jetson_ntc/config.json");
    ColorParams current(); // checks mtime and hot-reloads

private:
    std::string path_;
    std::chrono::system_clock::time_point last_mtime_{};
    ColorParams cached_; // defaults used if file missing/invalid

    bool exists_and_mtime(std::chrono::system_clock::time_point& mt) const;
    bool reload();
};

} // namespace icp
