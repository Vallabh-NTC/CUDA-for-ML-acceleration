#pragma once
#include <string>
#include "color_ops.cuh"

namespace pe {


class RuntimeControls {
public:
    explicit RuntimeControls(std::string path = "/home/moviemaker/editor.json");

    
    icp::ColorParams current();

private:
    std::string path_;
    struct MTime { long long sec{0}, nsec{0}; bool operator!=(const MTime& o) const { return sec!=o.sec || nsec!=o.nsec; } } last_{};
    icp::ColorParams cached_{};

    bool exists_and_mtime(MTime& mt) const;
    bool reload();
};

} // namespace pe
