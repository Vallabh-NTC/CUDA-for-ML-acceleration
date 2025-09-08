/**
 * @file rectify_config.hpp
 * @brief Configuration struct for rectification and color processing.
 *
 * Holds all tunable parameters used in the pipeline:
 *  - Geometry (fisheye → rectilinear projection).
 *  - Manual color controls (brightness, contrast, WB, etc).
 *  - Automatic controls (AE, AWB, tone mapping).
 *
 * Can be updated at runtime via JSON (RuntimeControls).
 */

#pragma once
#include <cstdint>

namespace icp {

struct RectifyConfig {
    // --- Geometry ---
    float fish_fov_deg = 195.1f; // fisheye lens FOV (deg)
    float out_hfov_deg = 90.0f;  // rectified horizontal FOV (deg)
    float cx_f = 959.50f;        // fisheye circle center X
    float cy_f = 539.50f;        // fisheye circle center Y
    float r_f  = 1100.77f;       // fisheye circle radius
    int   out_width = 1920;      // output width (rectified)

    // --- Manual color controls ---
    float brightness = 0.0f;   // Y offset
    float contrast   = 1.0f;   // scale around mid
    float saturation = 1.0f;   // chroma multiplier
    float gamma      = 1.0f;   // global gamma
    float wb_r = 1.0f, wb_g = 1.0f, wb_b = 1.0f; // manual WB multipliers

    // --- Automatic quality (Porsche preset) ---
    bool  auto_tone   = true;     // enable auto tone-mapping (histogram → LUT)
    bool  auto_wb     = true;     // enable auto white balance
    int   auto_roi_pct= 90;       // ROI percentage (of min side), 100=full frame
    int   auto_hist_step = 2;     // subsampling step for stats
    float auto_ae_step   = 0.08f; // AE smoothing step (speed of adaptation)

    float auto_lo_pct = 1.0f;     // low percentile for tone-mapping
    float auto_hi_pct = 99.0f;    // high percentile for tone-mapping
    float auto_tone_alpha = 0.10f;// histogram smoothing
    float auto_wb_alpha   = 0.05f;// WB smoothing
    float auto_wb_clamp   = 0.20f;// max WB per-frame change
    float auto_gamma_min  = 0.85f;// min gamma
    float auto_gamma_max  = 1.15f;// max gamma
    float target_Y        = 110.0f; // target luminance (mid-gray ~ pleasant)
};

} // namespace icp
