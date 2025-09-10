#include "runtime_controls.hpp"
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <regex>

namespace icp {

static bool read_text(const std::string& path, std::string& out)
{
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Minimal flat JSON: extract numbers/bools by key (no arrays, no nesting).
template<typename T>
static bool json_get_number(const std::string& s, const char* key, T& out)
{
    std::regex re(std::string("\"") + key + R"("\s*:\s*([-+]?\d+(\.\d+)?([eE][-+]?\d+)?))");
    std::smatch m;
    if (std::regex_search(s, m, re)) { out = static_cast<T>(std::stod(m[1])); return true; }
    return false;
}
static bool json_get_bool(const std::string& s, const char* key, bool& out)
{
    std::regex re(std::string("\"") + key + R"("\s*:\s*(true|false))");
    std::smatch m;
    if (std::regex_search(s, m, re)) { out = (m[1] == "true"); return true; }
    return false;
}

RuntimeControls::RuntimeControls(std::string path) : path_(std::move(path)) {}

bool RuntimeControls::exists_and_mtime(std::chrono::system_clock::time_point& mt) const
{
    struct stat st{};
    if (stat(path_.c_str(), &st) != 0) return false;
    mt = std::chrono::system_clock::from_time_t(st.st_mtime);
    return true;
}

bool RuntimeControls::reload()
{
    std::string txt;
    if (!read_text(path_, txt)) return false;

    ColorParams p = cached_; // start from current; override selectively

    json_get_bool(txt,  "enable",      p.enable);
    json_get_bool(txt,  "tv_range",    p.tv_range);

    json_get_number(txt, "exposure_ev", p.exposure_ev);
    json_get_number(txt, "contrast",    p.contrast);
    json_get_number(txt, "highlights",  p.highlights);
    json_get_number(txt, "shadows",     p.shadows);
    json_get_number(txt, "whites",      p.whites);
    json_get_number(txt, "gamma",       p.gamma);
    json_get_number(txt, "saturation",  p.saturation);

    cached_ = p;
    return true;
}

ColorParams RuntimeControls::current()
{
    std::chrono::system_clock::time_point mt;
    if (exists_and_mtime(mt)) {
        if (last_mtime_.time_since_epoch().count() == 0 || mt != last_mtime_) {
            if (reload()) last_mtime_ = mt;
        }
    }
    return cached_;
}

} // namespace icp
