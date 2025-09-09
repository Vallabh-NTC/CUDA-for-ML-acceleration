/**
 * @file rectify_config.hpp
 * @brief Configuration struct for fisheye rectification.
 *
 * Contains tunable parameters for fisheye → rectilinear projection.
 * Only geometry parameters are included here (no color settings).
 */

#pragma once
#include <cstdint>

namespace icp {

struct RectifyConfig {
    // --- Geometry parameters ---
    float fish_fov_deg = 195.1f; ///< fisheye lens field-of-view (degrees)
    float out_hfov_deg = 90.0f;  ///< rectified horizontal FOV (degrees)
    float cx_f = 959.50f;        ///< fisheye circle center X
    float cy_f = 539.50f;        ///< fisheye circle center Y
    float r_f  = 1100.77f;       ///< fisheye circle radius
    int   out_width = 1920;      ///< output width (rectified image)
};

} // namespace icp
