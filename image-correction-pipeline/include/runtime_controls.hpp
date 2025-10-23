#pragma once
#include <string>
#include <utility>      // for std::move
#include "color_ops.cuh"

namespace icp {

/**
 * Minimal file watcher + flat-or-sectioned JSON reader. No external deps.
 * - path_: JSON file path (default: /home/moviemaker/config.json).
 * - section_: "cam0", "cam1", "cam2" or "" (flat JSON mode).
 * - current(): checks mtime and hot-reloads on change, returns cached ColorParams.
 */
class RuntimeControls {
public:
    explicit RuntimeControls(std::string path = "/home/moviemaker/config.json",
                             std::string section = "");

    ColorParams current();

    void set_section(std::string section) { section_ = std::move(section); }
    const std::string& section() const { return section_; }

private:
    std::string path_;
    std::string section_; // "cam0"/"cam1"/"cam2" or empty for flat

    struct MTime {
        long long sec{0};
        long long nsec{0};
        bool operator!=(const MTime& o) const { return sec!=o.sec || nsec!=o.nsec; }
    } last_mtime_{};

    ColorParams cached_{};

    bool exists_and_mtime(MTime& mt) const;
    bool reload();
};

} // namespace icp
