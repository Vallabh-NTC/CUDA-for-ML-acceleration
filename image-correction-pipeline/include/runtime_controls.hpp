#pragma once
#include <string>
#include <chrono>
#include "color_ops.cuh"

namespace icp {

// Minimal file watcher + flat-or-sectioned JSON reader. No external deps.
class RuntimeControls {
public:
    // section: "cam0", "cam1", "cam2" or "" (flat JSON mode)
    explicit RuntimeControls(std::string path = "/home/jetson_ntc/config.json",
                             std::string section = "");

    // checks mtime (nanosecond precision when available) and hot-reloads
    ColorParams current();

    // change section at runtime (e.g., when binding instances)
    void set_section(std::string section) { section_ = std::move(section); }

private:
    std::string path_;
    std::string section_; // "cam0"/"cam1"/"cam2" or empty for flat
    // store both sec+nsec to detect sub-second updates
    struct MTime {
        long long sec{0};
        long long nsec{0};
        bool operator!=(const MTime& o) const { return sec!=o.sec || nsec!=o.nsec; }
    } last_mtime_{};

    ColorParams cached_; // defaults used if file missing/invalid

    bool exists_and_mtime(MTime& mt) const;
    bool reload();
};

} // namespace icp
