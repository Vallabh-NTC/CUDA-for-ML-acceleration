#pragma once

#include <vector>
#include <cstdint>
#include "kernel_draw_hand_points_nv12.cuh"  // for draw::Point2D

namespace gesture {

// Simple gesture types; you can extend this later.
enum class Type : int {
    NONE  = 0,
    PEACE = 1
};

// Result object for a single hand gesture classification.
struct Result {
    Type  type  = Type::NONE;
    float score = 0.0f;   // 0..1 approximate confidence of the rule
};

/**
 * Classify a single hand as "peace" (✌️) based on 2D keypoints.
 *
 * We assume a 21-keypoint layout similar to MediaPipe / trt_pose_hand:
 *   0: wrist
 *   1-4:  thumb   (cmc, mcp, ip, tip)
 *   5-8:  index   (mcp, pip, dip, tip)
 *   9-12: middle  (mcp, pip, dip, tip)
 *  13-16: ring    (mcp, pip, dip, tip)
 *  17-20: little  (mcp, pip, dip, tip)
 *
 * Rule for PEACE:
 *   - index  extended
 *   - middle extended
 *   - ring   bent
 *   - little bent
 *   - thumb  ignored (don't care)
 *
 * Input coordinates are in image pixels (draw::Point2D.x/y).
 * Conf is not strictly used yet, but you can plug it in later.
 */
Result classify_peace(const std::vector<draw::Point2D>& pts);

} // namespace gesture
