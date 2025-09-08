/**
 * @file runtime_controls.cpp
 * @brief Implementation of JSON hot-reload for runtime controls.
 *
 * Uses a background thread to monitor file changes (by modification time).
 * When the file is updated, it parses JSON and updates the active RectifyConfig.
 *
 * JSON format (example):
 * {
 *   "brightness": 0.0,
 *   "contrast": 1.2,
 *   "saturation": 1.1,
 *   "gamma": 1.0,
 *   "auto_tone": true,
 *   "auto_wb": true,
 *   "target_Y": 110.0
 * }
 *
 * Implementation details:
 *   - Simple std::ifstream + nlohmann/json (or custom parser).
 *   - Thread is joinable and stops in destructor.
 *   - Protects cfg_ with atomic updates (thread-safe reads).
 */

#include "runtime_controls.hpp"

#include <fstream>
#include <sstream>
#include <regex>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>

using namespace std;

static bool read_text_file(const std::string& path, std::string& out)
{
    std::ifstream ifs(path);
    if (!ifs.good()) return false;
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out = ss.str();
    return true;
}

static bool find_float(const std::string& text, const char* key, float& out)
{
    std::regex rx(std::string("\"") + key + "\"\\s*:\\s*([-+]?\\d+(?:\\.\\d+)?)");
    std::smatch m;
    if (std::regex_search(text, m, rx)) {
        try { out = std::stof(m[1].str()); return true; } catch (...) {}
    }
    return false;
}

// static
bool RuntimeControls::load_controls_from_file(const std::string& path, icp::RectifyConfig& rc)
{
    std::string text;
    if (!read_text_file(path, text)) return false;

    bool any = false; float v;

    if (find_float(text, "fish_fov_deg", v)) { rc.fish_fov_deg = v; any = true; }
    if (find_float(text, "out_hfov_deg", v)) { rc.out_hfov_deg = v; any = true; }
    if (find_float(text, "cx_f",        v)) { rc.cx_f         = v; any = true; }
    if (find_float(text, "cy_f",        v)) { rc.cy_f         = v; any = true; }
    if (find_float(text, "r_f",         v)) { rc.r_f          = v; any = true; }

    if (find_float(text, "brightness",  v)) { rc.brightness   = v; any = true; }
    if (find_float(text, "contrast",    v)) { rc.contrast     = v; any = true; }
    if (find_float(text, "saturation",  v)) { rc.saturation   = v; any = true; }
    if (find_float(text, "gamma",       v)) { rc.gamma        = v; any = true; }
    if (find_float(text, "wb_r",        v)) { rc.wb_r         = v; any = true; }
    if (find_float(text, "wb_g",        v)) { rc.wb_g         = v; any = true; }
    if (find_float(text, "wb_b",        v)) { rc.wb_b         = v; any = true; }

    return any;
}

RuntimeControls::RuntimeControls(std::string path, bool enable_watch)
: path_(std::move(path))
{
    // Prima lettura
    icp::RectifyConfig tmp = cfg_;
    load_controls_from_file(path_, tmp);
    { std::lock_guard<std::mutex> lk(mtx_); cfg_ = tmp; }

    // mtime iniziale (C++17)
    try {
        last_mtime_ = std::filesystem::last_write_time(path_);
    } catch (...) {
        last_mtime_ = std::filesystem::file_time_type{};
    }

    if (enable_watch) {
        th_ = std::thread(&RuntimeControls::watch_loop, this);
    }
}

RuntimeControls::~RuntimeControls()
{
    stop_.store(true);
    if (th_.joinable()) th_.join();
}

icp::RectifyConfig RuntimeControls::get() const
{
    std::lock_guard<std::mutex> lk(mtx_);
    return cfg_;
}

bool RuntimeControls::reload()
{
    icp::RectifyConfig tmp = get();
    if (!load_controls_from_file(path_, tmp)) return false;
    { std::lock_guard<std::mutex> lk(mtx_); cfg_ = tmp; }
    return true;
}

void RuntimeControls::watch_loop()
{
    using namespace std::chrono_literals;
    while (!stop_.load()) {
        std::this_thread::sleep_for(1s);
        try {
            auto now_mtime = std::filesystem::last_write_time(path_);
            if (now_mtime != last_mtime_) {
                last_mtime_ = now_mtime;
                if (reload()) {
                    std::cerr << "[runtime_controls] reloaded " << path_ << std::endl;
                }
            }
        } catch (...) {
            // file assente/non leggibile: ignora
        }
    }
}
