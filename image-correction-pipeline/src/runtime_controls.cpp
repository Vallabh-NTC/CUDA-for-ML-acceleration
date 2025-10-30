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

// ==== Tiny JSON helpers (regex-based, tolerant) ====
// Extract a top-level object by key: returns the {...} substring for "camX"
static bool json_extract_object(const std::string& s, const std::string& key, std::string& outObj)
{
    // Find '"key"' then first '{' then match braces
    std::regex rk("\"" + key + "\"\\s*:\\s*\\{");
    std::smatch m;
    if (!std::regex_search(s, m, rk)) return false;
    size_t start = m.position() + m.length() - 1; // points to '{'
    int depth = 0;
    for (size_t i = start; i < s.size(); ++i) {
        if (s[i] == '{') { if (depth++ == 0) start = i; }
        else if (s[i] == '}') { if (--depth == 0) { outObj = s.substr(start, i - start + 1); return true; } }
    }
    return false;
}

template<typename T>
static bool json_get_number(const std::string& s, const char* key, T& out)
{
    // Matches integers or floats, optionally in scientific notation.
    std::regex re(std::string("\"") + key + R"("\s*:\s*([-+]?\d+(\.\d+)?([eE][-+]?\d+)?))");
    std::smatch m; if (std::regex_search(s, m, re)) { out = static_cast<T>(std::stod(m[1])); return true; }
    return false;
}
static bool json_get_bool(const std::string& s, const char* key, bool& out)
{
    std::regex re(std::string("\"") + key + R"("\s*:\s*(true|false))");
    std::smatch m; if (std::regex_search(s, m, re)) { out = (m[1] == "true"); return true; }
    return false;
}

// -----------------------------------------------

RuntimeControls::RuntimeControls(std::string path, std::string section)
: path_(std::move(path)), section_(std::move(section)) {}

bool RuntimeControls::exists_and_mtime(MTime& mt) const
{
    struct stat st{};
    if (stat(path_.c_str(), &st) != 0) return false;

#if defined(__APPLE__)
    // macOS uses st_mtimespec
    mt.sec  = st.st_mtimespec.tv_sec;
    mt.nsec = st.st_mtimespec.tv_nsec;
#elif defined(_BSD_SOURCE) || defined(__FreeBSD__)
    mt.sec  = st.st_mtim.tv_sec;
    mt.nsec = st.st_mtim.tv_nsec;
#elif defined(_POSIX_C_SOURCE) && defined(__linux__)
    mt.sec  = st.st_mtim.tv_sec;
    mt.nsec = st.st_mtim.tv_nsec;
#else
    // Fallback: seconds only
    mt.sec  = st.st_mtime;
    mt.nsec = 0;
#endif
    return true;
}

bool RuntimeControls::reload()
{
    std::string txt;
    if (!read_text(path_, txt)) return false;

    // Scope used for camera color parameters (cam section or flat JSON)
    std::string scope = txt;
    if (!section_.empty()) {
        std::string sub;
        if (json_extract_object(txt, section_, sub)) scope = sub;
        // else: no section found -> keep using full text as a fallback
    }

    // Start from cached values; override selectively
    ColorParams p = cached_;

    // --- Camera (scoped) parameters ---
    json_get_bool(scope,  "enable",      p.enable);
    json_get_bool(scope,  "tv_range",    p.tv_range);

    json_get_number(scope, "exposure_ev", p.exposure_ev);
    json_get_number(scope, "contrast",    p.contrast);
    json_get_number(scope, "highlights",  p.highlights);
    json_get_number(scope, "shadows",     p.shadows);
    json_get_number(scope, "whites",      p.whites);
    json_get_number(scope, "gamma",       p.gamma);
    json_get_number(scope, "saturation",  p.saturation);

    // Added controls
    json_get_number(scope, "brightness",  p.brightness);
    json_get_number(scope, "brilliance",  p.brilliance);
    json_get_number(scope, "sharpness",   p.sharpness);

    // --- Global (root) AI flag ---
    // Read from the full JSON text (root), not from the camera section.
    // Accept both numeric 0/1 and boolean true/false for robustness.
    bool ai_ok = false;
    {
        int ai_num = 0;
        if (json_get_number(txt, "ai_enabled", ai_num)) {
            cached_ai_enabled_ = (ai_num != 0);
            ai_ok = true;
        } else {
            bool ai_bool = false;
            if (json_get_bool(txt, "ai_enabled", ai_bool)) {
                cached_ai_enabled_ = ai_bool;
                ai_ok = true;
            }
        }
        // If neither numeric nor boolean found, keep previous cached_ai_enabled_
        // (default remains false until the key appears).
        (void)ai_ok;
    }

    cached_ = p;
    return true;
}

ColorParams RuntimeControls::current()
{
    MTime mt;
    if (exists_and_mtime(mt)) {
        if ((last_mtime_.sec == 0 && last_mtime_.nsec == 0) || mt != last_mtime_) {
            if (reload()) last_mtime_ = mt;
        }
    }
    return cached_;
}

} // namespace icp
