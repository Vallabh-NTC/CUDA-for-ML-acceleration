#pragma once
#include <gst/gst.h>
#include <string>
#include <cstdint>

// Minimal, no prints. Returns bool for success/failure.
class GStreamerCamera {
public:
    explicit GStreamerCamera(std::string pipeline_desc);
    ~GStreamerCamera();

    // Start the pipeline (expects an appsink named "mysink" producing RGBA frames).
    bool start();

    // Pull one frame into pinned, host-mapped memory.
    // On success:
    //  - *device_ptr  -> device alias of pinned host buffer (valid while this object lives)
    //  - *host_ptr    -> host pointer to the same pinned buffer
    //  - *num_bytes   -> buffer size in bytes
    //  - width/height -> optional; filled if non-null
    bool grab_frame_to_pinned(uint8_t** device_ptr,
                              uint8_t** host_ptr,
                              size_t*   num_bytes,
                              int*      width  = nullptr,
                              int*      height = nullptr,
                              guint64   timeout_ns = 2ULL * 1000 * 1000 * 1000); // 2s

    // Stop/teardown.
    void stop();

    // Non-copyable
    GStreamerCamera(const GStreamerCamera&) = delete;
    GStreamerCamera& operator=(const GStreamerCamera&) = delete;

private:
    bool ensure_pinned_capacity(size_t bytes);
    void reset_pipeline();

    std::string pipeline_desc_;
    GstElement* pipeline_{nullptr};
    GstElement* sink_elem_{nullptr};   // appsink element
    GstBus*     bus_{nullptr};

    // Pinned host-mapped buffer we reuse/grow
    uint8_t* pinned_host_{nullptr};
    size_t   pinned_size_{0};
};
