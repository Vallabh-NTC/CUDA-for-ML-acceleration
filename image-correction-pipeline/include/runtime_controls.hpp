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
 *
 * NEW:
 * - Global flag "ai_enabled" (0/1) is read from the JSON root (not from camera sections).
 *   Default = 0 (disabled). Access via ai_enabled().
 */
class RuntimeControls {
public:
    explicit RuntimeControls(std::string path = "/home/moviemaker/config.json",
                             std::string section = "");

    // Returns current color parameters (possibly hot-reloaded).
    ColorParams current();

    // Camera section getter/setter ("cam0"/"cam1"/"cam2" or empty for flat JSON).
    void set_section(std::string section) { section_ = std::move(section); }
    const std::string& section() const { return section_; }

    // Global AI enable flag (0/1) read from JSON root, default false.
    bool ai_enabled() const { return cached_ai_enabled_; }

private:
    std::string path_;
    std::string section_; // "cam0"/"cam1"/"cam2" or empty for flat

    struct MTime {
        long long sec{0};
        long long nsec{0};
        bool operator!=(const MTime& o) const { return sec!=o.sec || nsec!=o.nsec; }
    } last_mtime_{};

    // Cached camera color controls
    ColorParams cached_{};

    // Cached global AI flag (from JSON root). Default = false (0).
    bool cached_ai_enabled_{false};

    // FS + parsing helpers
    bool exists_and_mtime(MTime& mt) const;
    bool reload();
};

} // namespace icp
