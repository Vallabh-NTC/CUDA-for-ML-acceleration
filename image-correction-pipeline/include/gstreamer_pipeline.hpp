#pragma once
#include <gst/gst.h>
#include <string>

class GStreamerCamera {
public:
    explicit GStreamerCamera(std::string pipeline_desc);
    ~GStreamerCamera();

    // Runs the pipeline (blocking) until EOS or ERROR.
    // Returns true on clean EOS, false on error/early failure.
    bool run();

private:
    std::string pipeline_desc_;
    GstElement* pipeline_{nullptr};
    GstBus* bus_{nullptr};

    // non-copyable
    GStreamerCamera(const GStreamerCamera&) = delete;
    GStreamerCamera& operator=(const GStreamerCamera&) = delete;
};
