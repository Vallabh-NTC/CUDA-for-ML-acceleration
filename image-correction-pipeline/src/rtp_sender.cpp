#include "rtp_sender.hpp"
#include <sstream>
#include <cstring>

RtpSender::RtpSender(const std::string& host, int port,
                     int width, int height, int fps_num, int fps_den)
    : host_(host), port_(port), w_(width), h_(height),
      fps_num_(fps_num), fps_den_(fps_den) {}

RtpSender::~RtpSender() { stop(); }

std::string RtpSender::build_pipeline_str() const {
    // Caps for incoming RGBA frames
    std::ostringstream caps;
    caps << "video/x-raw,format=RGBA,width=" << w_ << ",height=" << h_
         << ",framerate=" << fps_num_ << "/" << fps_den_;

    // Tuned for robustness + low latency:
    // - nvvidconv -> NV12 (NVMM) for NVENC
    // - H.264 High profile (profile=2)
    // - IDR every 15 frames (~0.5s @30fps) -> faster recovery
    // - bitrate 16 Mbps (raise if needed)
    // - rtph264pay config-interval=-1 (send SPS/PPS at each IDR)
    // - mtu=1200 and qos=false on udpsink
    std::ostringstream ss;
   ss
    << "appsrc name=rectsrc is-live=true format=time block=true do-timestamp=true caps=\"" << caps.str() << "\" ! "
    << "queue max-size-buffers=8 max-size-time=0 max-size-bytes=0 leaky=downstream ! "
    << "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
    << "nvv4l2h264enc profile=2 insert-sps-pps=1 iframeinterval=30 control-rate=1 bitrate=9000000 preset-level=1 ! "
    << "h264parse ! rtph264pay config-interval=-1 pt=96 mtu=1000 ! "
    << "udpsink host=" << host_ << " port=" << port_
    << " sync=false async=false qos=false buffer-size=2097152";

    return ss.str();
}

bool RtpSender::start() {
    if (started_) return true;

    GError* err = nullptr;
    std::string desc = build_pipeline_str();
    pipeline_ = gst_parse_launch(desc.c_str(), &err);
    if (!pipeline_) {
        if (err) g_error_free(err);
        return false;
    }

    appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "rectsrc");
    if (!appsrc_) {
        gst_object_unref(pipeline_); pipeline_ = nullptr;
        return false;
    }

    // Configure appsrc for live streaming with timestamps
    g_object_set(G_OBJECT(appsrc_),
                 "stream-type", 0,           // GST_APP_STREAM_TYPE_STREAM
                 "format", GST_FORMAT_TIME,
                 "is-live", TRUE,
                 "do-timestamp", TRUE,
                 NULL);

    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        gst_object_unref(appsrc_); appsrc_ = nullptr;
        gst_object_unref(pipeline_); pipeline_ = nullptr;
        return false;
    }
    started_ = true;
    frame_count_ = 0;
    return true;
}

void RtpSender::stop() {
    if (!started_) return;
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    if (appsrc_) { gst_object_unref(appsrc_); appsrc_ = nullptr; }
    if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
    started_ = false;
}

bool RtpSender::push_rgba(const uint8_t* rgba, int width, int height, int stride_bytes) {
    if (!started_ || !appsrc_) return false;
    if (width != w_ || height != h_) return false;

    const int row_bytes = width * 4;
    const gsize size = (gsize)row_bytes * height;

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    if (!buffer) return false;

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return false;
    }

    // Copy row-by-row if stride differs
    uint8_t* dst = map.data;
    if (stride_bytes == row_bytes) {
        std::memcpy(dst, rgba, size);
    } else {
        for (int y = 0; y < height; ++y) {
            std::memcpy(dst + (size_t)y * row_bytes,
                        rgba + (size_t)y * stride_bytes,
                        row_bytes);
        }
    }
    gst_buffer_unmap(buffer, &map);

    // Timestamp assuming constant FPS
    const GstClockTime frame_duration = gst_util_uint64_scale_int(GST_SECOND, fps_den_, fps_num_);
    GST_BUFFER_PTS(buffer) = frame_count_ * frame_duration;
    GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
    GST_BUFFER_DURATION(buffer) = frame_duration;
    frame_count_++;

    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
    return (ret == GST_FLOW_OK);
}
