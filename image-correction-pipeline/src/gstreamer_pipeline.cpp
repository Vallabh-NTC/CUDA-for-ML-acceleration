#include "gstreamer_pipeline.hpp"
#include <gst/app/gstappsink.h>
#include <cuda_runtime.h>
#include <cstring>

static inline bool cuda_ok(cudaError_t e) { return e == cudaSuccess; }

GStreamerMultiCamera::GStreamerMultiCamera(std::string pipeline_desc,
                                           std::vector<std::string> sink_names)
    : pipeline_desc_(std::move(pipeline_desc)),
      sink_names_(std::move(sink_names)) {}

GStreamerMultiCamera::~GStreamerMultiCamera() {
    stop();
    for (size_t i = 0; i < pinned_host_.size(); ++i) {
        if (pinned_host_[i]) cudaFreeHost(pinned_host_[i]);
    }
    pinned_host_.clear();
    pinned_size_.clear();
}

void GStreamerMultiCamera::reset_pipeline() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        if (bus_) { gst_object_unref(bus_); bus_ = nullptr; }
        for (auto*& s : sink_elems_) {
            if (s) { gst_object_unref(s); s = nullptr; }
        }
        sink_elems_.clear();
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
}

bool GStreamerMultiCamera::start() {
    reset_pipeline();

    if (sink_names_.empty()) return false;

    // allocate pinned vectors (lazy alloc per cam)
    pinned_host_.assign(sink_names_.size(), nullptr);
    pinned_size_.assign(sink_names_.size(), 0);

    GError* err = nullptr;
    pipeline_ = gst_parse_launch(pipeline_desc_.c_str(), &err);
    if (!pipeline_) {
        if (err) g_error_free(err);
        return false;
    }
    bus_ = gst_element_get_bus(pipeline_);

    // bind all appsinks by name
    sink_elems_.resize(sink_names_.size(), nullptr);
    for (size_t i = 0; i < sink_names_.size(); ++i) {
        sink_elems_[i] = gst_bin_get_by_name(GST_BIN(pipeline_), sink_names_[i].c_str());
        if (!sink_elems_[i]) {
            reset_pipeline();
            return false;
        }
    }

    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        reset_pipeline();
        return false;
    }
    return true;
}

bool GStreamerMultiCamera::ensure_pinned_capacity(int i, size_t bytes) {
    if (i < 0 || i >= (int)pinned_host_.size()) return false;
    if (bytes <= pinned_size_[i]) return true;
    if (pinned_host_[i]) {
        cudaFreeHost(pinned_host_[i]);
        pinned_host_[i] = nullptr;
        pinned_size_[i] = 0;
    }
    // Caller must have enabled cudaDeviceMapHost beforehand
    if (!cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&pinned_host_[i]), bytes, cudaHostAllocMapped)))
        return false;
    pinned_size_[i] = bytes;
    return true;
}

bool GStreamerMultiCamera::grab_frame_to_pinned(int i,
                                                uint8_t** device_ptr,
                                                uint8_t** host_ptr,
                                                size_t*   num_bytes,
                                                int*      width,
                                                int*      height,
                                                guint64   timeout_ns)
{
    if (!pipeline_ || i < 0 || i >= (int)sink_elems_.size()) return false;

    GstAppSink* appsink = GST_APP_SINK(sink_elems_[i]);
    GstSample* sample = gst_app_sink_try_pull_sample(appsink, timeout_ns);
    if (!sample) return false;

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    const GstStructure* s = gst_caps_get_structure(caps, 0);

    int w = 0, h = 0;
    gst_structure_get_int(s, "width", &w);
    gst_structure_get_int(s, "height", &h);
    const char* fmt = gst_structure_get_string(s, "format");
    if (!fmt || std::string(fmt) != "RGBA") {
        gst_sample_unref(sample);
        return false;
    }

    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return false;
    }

    if (!ensure_pinned_capacity(i, map.size)) {
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return false;
    }
    std::memcpy(pinned_host_[i], map.data, map.size);

    void* d_alias = nullptr;
    if (!cuda_ok(cudaHostGetDevicePointer(&d_alias, pinned_host_[i], 0))) {
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return false;
    }

    if (device_ptr) *device_ptr = static_cast<uint8_t*>(d_alias);
    if (host_ptr)   *host_ptr   = pinned_host_[i];
    if (num_bytes)  *num_bytes  = map.size;
    if (width)      *width      = w;
    if (height)     *height     = h;

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    return true;
}

void GStreamerMultiCamera::stop() {
    reset_pipeline();
}
