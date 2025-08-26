#include "gstreamer_pipeline.hpp"
#include <iostream>

GStreamerCamera::GStreamerCamera(std::string pipeline_desc)
    : pipeline_desc_(std::move(pipeline_desc)) {
    // Caller should ensure gst_init() was called before constructing.
}

GStreamerCamera::~GStreamerCamera() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        if (bus_) gst_object_unref(bus_);
        gst_object_unref(pipeline_);
    }
}

bool GStreamerCamera::run() {
    GError* err = nullptr;
    pipeline_ = gst_parse_launch(pipeline_desc_.c_str(), &err);
    if (!pipeline_) {
        std::cerr << "Failed to create pipeline: "
                  << (err ? err->message : "unknown error") << std::endl;
        if (err) g_error_free(err);
        return false;
    }

    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Failed to set pipeline to PLAYING" << std::endl;
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return false;
    }

    bus_ = gst_element_get_bus(pipeline_);
    bool running = true;
    bool ok = true;

    while (running) {
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus_, GST_CLOCK_TIME_NONE,
            static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)
        );

        if (msg != nullptr) {
            switch (GST_MESSAGE_TYPE(msg)) {
                case GST_MESSAGE_ERROR: {
                    GError* e = nullptr;
                    gchar* dbg = nullptr;
                    gst_message_parse_error(msg, &e, &dbg);
                    std::cerr << "Error from element "
                              << GST_OBJECT_NAME(msg->src) << ": "
                              << (e ? e->message : "unknown") << std::endl;
                    if (dbg) {
                        std::cerr << "Debug info: " << dbg << std::endl;
                        g_free(dbg);
                    }
                    if (e) g_error_free(e);
                    ok = false;
                    running = false;
                    break;
                }
                case GST_MESSAGE_EOS:
                    std::cout << "End-of-stream" << std::endl;
                    running = false;
                    break;
                default:
                    break;
            }
            gst_message_unref(msg);
        }
    }

    gst_element_set_state(pipeline_, GST_STATE_NULL);
    return ok;
}
