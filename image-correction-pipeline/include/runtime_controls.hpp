#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>

// Runtime-adjustable color controls (thread-safe via atomics).
struct RuntimeControls {
    std::atomic<float> brightness{0.0f};
    std::atomic<float> contrast{1.0f};
    std::atomic<float> saturation{1.0f};
    std::atomic<float> gamma{1.0f};
    std::atomic<float> wb_r{1.0f}, wb_g{1.0f}, wb_b{1.0f};

    // Take a plain snapshot (no atomics in kernels).
    void snapshot(float& out_bri, float& out_con, float& out_sat,
                  float& out_gamma, float& out_wbr, float& out_wbg, float& out_wbb) const {
        out_bri = brightness.load(std::memory_order_relaxed);
        out_con = contrast.load(std::memory_order_relaxed);
        out_sat = saturation.load(std::memory_order_relaxed);
        out_gamma = gamma.load(std::memory_order_relaxed);
        out_wbr = wb_r.load(std::memory_order_relaxed);
        out_wbg = wb_g.load(std::memory_order_relaxed);
        out_wbb = wb_b.load(std::memory_order_relaxed);
    }
};

// Tiny tolerant parser for a flat JSON controls file.
// Keys: brightness, contrast, saturation, gamma, wb_r, wb_g, wb_b
inline bool load_controls_from_file(const std::string& path, RuntimeControls& rc) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (len <= 0 || len > (1<<20)) { std::fclose(f); return false; } // 1MB guard
    std::string buf; buf.resize((size_t)len);
    size_t rd = std::fread(buf.data(), 1, (size_t)len, f);
    std::fclose(f);
    if (rd != (size_t)len) return false;

    auto find_val = [&](const char* key, float& out){
        const char* p = std::strstr(buf.c_str(), key);
        if (!p) return false;
        p = std::strchr(p, ':');
        if (!p) return false;
        ++p;
        while (*p==' ' || *p=='\t') ++p;
        char* end=nullptr;
        float v = std::strtof(p, &end);
        if (p==end) return false;
        out = v; return true;
    };

    float v;
    if (find_val("brightness", v)) rc.brightness.store(v, std::memory_order_relaxed);
    if (find_val("contrast",   v)) rc.contrast.store(v, std::memory_order_relaxed);
    if (find_val("saturation", v)) rc.saturation.store(v, std::memory_order_relaxed);
    if (find_val("gamma",      v)) rc.gamma.store(v, std::memory_order_relaxed);
    if (find_val("wb_r",       v)) rc.wb_r.store(v, std::memory_order_relaxed);
    if (find_val("wb_g",       v)) rc.wb_g.store(v, std::memory_order_relaxed);
    if (find_val("wb_b",       v)) rc.wb_b.store(v, std::memory_order_relaxed);
    return true;
}

// Simple polling watcher: reload when file timestamp changes.
inline void watch_controls_file(const std::string& path,
                                RuntimeControls& rc,
                                std::atomic<bool>& running,
                                std::chrono::milliseconds period = std::chrono::milliseconds(200))
{
    namespace fs = std::filesystem;
    fs::file_time_type last{};
    while (running.load(std::memory_order_relaxed)) {
        std::error_code ec;
        auto cur = fs::last_write_time(path, ec);
        if (!ec && cur != fs::file_time_type{} && cur != last) {
            load_controls_from_file(path, rc);
            last = cur;
        }
        std::this_thread::sleep_for(period);
    }
}
