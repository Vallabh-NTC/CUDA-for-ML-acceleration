// image-correction-pipeline/include/rectify_config.hpp
#pragma once

// ----------------------------------------------------------------------------
// RectifyConfig
// ----------------------------------------------------------------------------
// Holds ALL parameters used by the CUDA kernel:
//   1) Fisheye-to-perspective mapping parameters for an equidistant fisheye
//      model (typical for wide FOV lenses).
//   2) Color controls applied in linear 8-bit space (except gamma which is
//      applied on normalized [0..1]) to allow runtime tuning without
//      recompilation or pipeline restart.
//
// IMPORTANT FORMAT ASSUMPTIONS (handled by the plugin/pipeline):
// - The CUDA kernel expects pitch-linear ABGR (which corresponds to RGBA byte
//   order stored in an ABGR enum; this is how CU_EGL_COLOR_FORMAT_ABGR is
//   reported by nvivafilter).
// - The pipeline MUST convert from camera output (often UYVY/NV12) into RGBA
//   in NVMM *before* nvivafilter (e.g., with `nvvidconv ! ... format=RGBA`).
// ----------------------------------------------------------------------------
struct RectifyConfig {
    // --- Equidistant fisheye model parameters (all in pixels or degrees) ---
    // Horizontal fisheye FOV (degrees). Controls how quickly rays diverge radially.
    float fish_fov_deg = 195.1f;

    // Desired perspective horizontal FOV (degrees). Affects fx.
    float out_hfov_deg = 90.0f;

    // Fisheye circle optical center in source image (pixels).
    float cx_f = 959.50f;
    float cy_f = 539.50f;

    // Fisheye circle radius (pixels). Anything beyond this is considered invalid.
    float r_f  = 1100.77f;

    // --- Color pipeline parameters (linear 8-bit domain unless noted) ---
    // Brightness is an additive bias; Contrast is a multiplicative gain.
    float brightness = 0.0f;   // add per channel in [0..255]
    float contrast   = 1.0f;   // >1 increases contrast, <1 reduces

    // Saturation is computed relative to luma (Rec.709). 0=grayscale, 1=neutral.
    float saturation = 1.0f;

    // Gamma is applied LAST on normalized [0..1]. gamma < 1 brightens mid-tones.
    float gamma      = 1.0f;

    // Simple per-channel white balance multipliers (pre-gamma).
    float wb_r = 1.0f;
    float wb_g = 1.0f;
    float wb_b = 1.0f;
};
