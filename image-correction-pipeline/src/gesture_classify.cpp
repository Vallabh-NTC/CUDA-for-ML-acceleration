#include "gesture_classify.hpp"

#include <cmath>

namespace {

// Compute angle (in degrees) between two 2D vectors.
inline float angle_between(float ax, float ay,
                           float bx, float by)
{
    float dot = ax * bx + ay * by;
    float na  = std::sqrt(ax * ax + ay * ay);
    float nb  = std::sqrt(bx * bx + by * by);
    if (na < 1e-6f || nb < 1e-6f) {
        // Degenerate, treat as fully bent
        return 180.0f;
    }
    float c = dot / (na * nb);
    if (c < -1.0f) c = -1.0f;
    if (c >  1.0f) c =  1.0f;
    return std::acos(c) * 180.0f / static_cast<float>(M_PI);
}

// Helper: is a single finger extended, given its 4 joints?
// The finger is defined by 4 points (in order from base to tip):
//   a = base (e.g. MCP)
//   b = joint1 (PIP)
//   c = joint2 (DIP)
//   d = tip
//
// We check the angles between segments:
//
//   a---b---c---d
//   v1 = b-a, v2 = c-b, v3 = d-c
//
// If both angles (v1,v2) and (v2,v3) are small, the finger is straight.
inline bool is_finger_extended(const draw::Point2D& a,
                               const draw::Point2D& b,
                               const draw::Point2D& c,
                               const draw::Point2D& d)
{
    // If you want to use confidences, you can gate here:
    // if (a.conf < 0.2f || b.conf < 0.2f || c.conf < 0.2f || d.conf < 0.2f)
    //     return false;

    float v1x = static_cast<float>(b.x - a.x);
    float v1y = static_cast<float>(b.y - a.y);
    float v2x = static_cast<float>(c.x - b.x);
    float v2y = static_cast<float>(c.y - b.y);
    float v3x = static_cast<float>(d.x - c.x);
    float v3y = static_cast<float>(d.y - c.y);

    float a1 = angle_between(v1x, v1y, v2x, v2y);
    float a2 = angle_between(v2x, v2y, v3x, v3y);

    // Thresholds can be tuned; smaller = stricter "straight" definition.
    const float kStraightThreshDeg = 23.0f;

    return (a1 < kStraightThreshDeg && a2 < kStraightThreshDeg);
}

} // anonymous namespace

namespace gesture {

Result classify_peace(const std::vector<draw::Point2D>& pts)
{
    Result r;

    // We expect at least 21 points in MediaPipe / trt_pose_hand layout.
    if (pts.size() < 21) {
        r.type  = Type::NONE;
        r.score = 0.0f;
        return r;
    }

    // Finger mapping (index in pts):
    //
    // thumb : 1-4  (not used for peace)
    // index : 5-8
    // middle: 9-12
    // ring  : 13-16
    // little: 17-20

    bool index_ext   = is_finger_extended(pts[5],  pts[6],  pts[7],  pts[8]);
    bool middle_ext  = is_finger_extended(pts[9],  pts[10], pts[11], pts[12]);
    bool ring_ext    = is_finger_extended(pts[13], pts[14], pts[15], pts[16]);
    bool little_ext  = is_finger_extended(pts[17], pts[18], pts[19], pts[20]);

    bool ring_bent   = !ring_ext;
    bool little_bent = !little_ext;

    // Build a weighted score:
    //   - index & middle extended are more important
    //   - ring & little bent still contribute, but less
    //
    // Weights:
    //   index_ext   → +0.35
    //   middle_ext  → +0.35
    //   ring_bent   → +0.15
    //   little_bent → +0.15
    //
    // Max = 1.0 for a "perfect" peace sign.
    float score = 0.0f;
    if (index_ext)   score += 0.35f;
    if (middle_ext)  score += 0.35f;
    if (ring_bent)   score += 0.15f;
    if (little_bent) score += 0.15f;

    r.score = score;

    // Require all four conditions to be true to call it PEACE.
    // The external code can then apply a score threshold (e.g. 0.8)
    // to be even stricter.
    if (index_ext && middle_ext && ring_bent && little_bent) {
        r.type = Type::PEACE;
    } else {
        r.type = Type::NONE;
    }

    return r;
}

} // namespace gesture
