// image-correction-pipeline/src/runtime_controls.cpp
#include "runtime_controls.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

// ----------------------------------------------------------------------------
// load_controls_from_file
// ----------------------------------------------------------------------------
// - Reads the entire file into memory (upper bound 1 MB for safety).
// - Very tolerant parsing: looks for "key: value" pairs and extracts floats.
// - Updates only keys that appear; missing keys leave existing values intact.
// - Returns 'true' if parsing *attempt* succeeded (file read and at least
//   parse attempt). We still return true if some keys are missing.
// ----------------------------------------------------------------------------
bool RuntimeControls::load_controls_from_file(const std::string& path, RectifyConfig& rc) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (len <= 0 || len > (1<<20)) { std::fclose(f); return false; } // <= 1 MB guard
    std::string buf(len, '\0');
    size_t rd = std::fread(buf.data(), 1, (size_t)len, f);
    std::fclose(f);
    if (rd != (size_t)len) return false;

    // Tiny parser: find "key : <float>" using strstr/strchr/strtof
    auto find_val=[&](const char* key,float& out){
        const char* p = std::strstr(buf.c_str(), key);
        if(!p) return false;
        p = std::strchr(p, ':'); if(!p) return false; ++p;
        while(*p==' '||*p=='\t') ++p;
        char* end=nullptr;
        float v = std::strtof(p,&end);
        if(p==end) return false;
        out = v; return true;
    };

    float v;
    // Optics / mapping
    if(find_val("fish_fov_deg", v)) rc.fish_fov_deg = v;
    if(find_val("out_hfov_deg", v)) rc.out_hfov_deg = v;
    if(find_val("cx_f",        v)) rc.cx_f = v;
    if(find_val("cy_f",        v)) rc.cy_f = v;
    if(find_val("r_f",         v)) rc.r_f  = v;

    // Color
    if(find_val("brightness",  v)) rc.brightness = v;
    if(find_val("contrast",    v)) rc.contrast   = v;
    if(find_val("saturation",  v)) rc.saturation = v;
    if(find_val("gamma",       v)) rc.gamma      = v;
    if(find_val("wb_r",        v)) rc.wb_r       = v;
    if(find_val("wb_g",        v)) rc.wb_g       = v;
    if(find_val("wb_b",        v)) rc.wb_b       = v;

    return true;
}

RuntimeControls::RuntimeControls(const std::string& path): path_(path) {
    running_.store(true, std::memory_order_relaxed);

    // Initial best-effort load (if JSON exists)
    RectifyConfig tmp;
    load_controls_from_file(path_, tmp);
    cfg_ = tmp;

    // Launch background watcher
    watcher_ = std::thread([this]{ watch_loop(); });
}

RuntimeControls::~RuntimeControls() {
    running_.store(false, std::memory_order_relaxed);
    if (watcher_.joinable()) watcher_.join();
}

void RuntimeControls::watch_loop() {
    namespace fs = std::filesystem;
    fs::file_time_type last{};
    while (running_.load(std::memory_order_relaxed)) {
        std::error_code ec;
        // mtime check is cheap; avoids unnecessary reads
        auto cur = fs::last_write_time(path_, ec);
        if (!ec && cur != fs::file_time_type{} && cur != last) {
            RectifyConfig tmp = cfg_;
            if (load_controls_from_file(path_, tmp)) {
                // single POD assignment: safe w/o extra locking
                cfg_ = tmp;
            }
            last = cur;
        }
        std::this_thread::sleep_for(200ms);
    }
}
