#pragma once
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <string>
#include <cstdint>

/**
 * RTP sender for RGBA frames using Jetson H.264 HW encoder over UDP.
 *
 * Pipeline:
 *   appsrc (RGBA, system mem)
 *   -> queue (leaky)
 *   -> nvvidconv
 *   -> video/x-raw(memory:NVMM),format=NV12
 *   -> nvv4l2h264enc (High profile, low-latency settings)
 *   -> h264parse
 *   -> rtph264pay config-interval=1 pt=96
 *   -> udpsink host=... port=...
 */
class RtpSender {
public:
    RtpSender(const std::string& host, int port,
              int width, int height, int fps_num, int fps_den);
    ~RtpSender();

    bool start();
    void stop();

    // Push one RGBA frame (stride in bytes). The data is copied into a GstBuffer.
    bool push_rgba(const uint8_t* rgba, int width, int height, int stride_bytes);

private:
    std::string host_;
    int port_;
    int w_, h_, fps_num_, fps_den_;

    GstElement *pipeline_ = nullptr;
    GstElement *appsrc_   = nullptr;

    bool started_ = false;
    guint64 frame_count_ = 0;

    std::string build_pipeline_str() const;
};
