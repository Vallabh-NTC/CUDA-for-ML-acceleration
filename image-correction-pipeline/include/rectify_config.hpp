/**
 * @file rectify_config.hpp
 * @brief Configuration struct for rectification only.
 *
 * Holds tunable parameters for fisheye → rectilinear projection.
 * This stripped-down version removes all color/AE/AWB/tone settings.
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
};

} // namespace icp
