#pragma once
#include <gst/gst.h>
#include <string>
#include <vector>
#include <cstdint>

// Manages ONE pipeline string that contains multiple parallel branches,
// each ending in a unique appsink (e.g., mysink0, mysink1, ...).
class GStreamerMultiCamera {
public:
    // pipeline_desc must define all branches + unique appsink names (size = num_cams).
    GStreamerMultiCamera(std::string pipeline_desc,
                         std::vector<std::string> sink_names);
    ~GStreamerMultiCamera();

    bool start();   // set pipeline to PLAYING and bind all sinks
    void stop();    // teardown

    // Pull one frame from appsink[i] into pinned, host-mapped memory.
    // Returns device alias (mapped) + host ptr + size + width/height.
    bool grab_frame_to_pinned(int i,
                              uint8_t** device_ptr,
                              uint8_t** host_ptr,
                              size_t*   num_bytes,
                              int*      width  = nullptr,
                              int*      height = nullptr,
                              guint64   timeout_ns = 2ULL * 1000 * 1000 * 1000); // 2s

    int num_cams() const { return static_cast<int>(sink_names_.size()); }

    // Non-copyable
    GStreamerMultiCamera(const GStreamerMultiCamera&) = delete;
    GStreamerMultiCamera& operator=(const GStreamerMultiCamera&) = delete;

private:
    bool ensure_pinned_capacity(int i, size_t bytes);
    void reset_pipeline();

    std::string pipeline_desc_;
    std::vector<std::string> sink_names_;

    GstElement* pipeline_{nullptr};
    GstBus*     bus_{nullptr};
    std::vector<GstElement*> sink_elems_; // appsinks in same order as sink_names_

    // per-camera pinned host buffers
    std::vector<uint8_t*> pinned_host_;
    std::vector<size_t>   pinned_size_;
};
