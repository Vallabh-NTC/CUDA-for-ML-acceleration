#include "gstreamer_pipeline.hpp"
#include <gst/app/gstappsink.h>
#include <cuda_runtime.h>
#include <cstring> // std::memcpy

static inline bool cuda_ok(cudaError_t e) { return e == cudaSuccess; }

GStreamerCamera::GStreamerCamera(std::string pipeline_desc)
    : pipeline_desc_(std::move(pipeline_desc)) {}

GStreamerCamera::~GStreamerCamera() {
    stop();
    if (pinned_host_) {
        cudaFreeHost(pinned_host_);
        pinned_host_ = nullptr;
        pinned_size_ = 0;
    }
}

void GStreamerCamera::reset_pipeline() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        if (bus_)      { gst_object_unref(bus_);      bus_ = nullptr; }
        if (sink_elem_){ gst_object_unref(sink_elem_); sink_elem_ = nullptr; }
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
}

bool GStreamerCamera::start() {
    reset_pipeline();

    GError* err = nullptr;
    pipeline_ = gst_parse_launch(pipeline_desc_.c_str(), &err);
    if (!pipeline_) {
        if (err) g_error_free(err);
        return false;
    }
    bus_ = gst_element_get_bus(pipeline_);

    // Expect an appsink named "mysink" in the pipeline descriptor
    sink_elem_ = gst_bin_get_by_name(GST_BIN(pipeline_), "mysink");
    if (!sink_elem_) {
        reset_pipeline();
        return false;
    }

    // Play
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        reset_pipeline();
        return false;
    }
    return true;
}

bool GStreamerCamera::ensure_pinned_capacity(size_t bytes) {
    if (bytes <= pinned_size_) return true;
    if (pinned_host_) {
        cudaFreeHost(pinned_host_);
        pinned_host_ = nullptr;
        pinned_size_ = 0;
    }
    // caller must have set cudaDeviceFlags(cudaDeviceMapHost) before
    if (!cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&pinned_host_), bytes, cudaHostAllocMapped)))
        return false;
    pinned_size_ = bytes;
    return true;
}

bool GStreamerCamera::grab_frame_to_pinned(uint8_t** device_ptr,
                                           uint8_t** host_ptr,
                                           size_t*   num_bytes,
                                           int*      width,
                                           int*      height,
                                           guint64   timeout_ns) {
    if (!pipeline_ || !sink_elem_) return false;

    GstAppSink* appsink = GST_APP_SINK(sink_elem_);
    GstSample* sample = gst_app_sink_try_pull_sample(appsink, timeout_ns);
    if (!sample) return false;

    // Extract buffer & caps
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    const GstStructure* s = gst_caps_get_structure(caps, 0);

    int w = 0, h = 0;
    gst_structure_get_int(s, "width", &w);
    gst_structure_get_int(s, "height", &h);
    const char* fmt = gst_structure_get_string(s, "format");
    // We expect RGBA (8-bit)
    if (!fmt || std::string(fmt) != "RGBA") {
        gst_sample_unref(sample);
        return false;
    }

    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return false;
    }

    // Ensure pinned capacity and copy data
    if (!ensure_pinned_capacity(map.size)) {
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return false;
    }
    std::memcpy(pinned_host_, map.data, map.size);

    // Device alias to the same pinned host memory
    void* d_alias = nullptr;
    if (!cuda_ok(cudaHostGetDevicePointer(&d_alias, pinned_host_, 0))) {
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return false;
    }

    // Output values
    if (device_ptr) *device_ptr = static_cast<uint8_t*>(d_alias);
    if (host_ptr)   *host_ptr   = pinned_host_;
    if (num_bytes)  *num_bytes  = map.size;
    if (width)      *width      = w;
    if (height)     *height     = h;

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    return true;
}

void GStreamerCamera::stop() {
    reset_pipeline();
}
