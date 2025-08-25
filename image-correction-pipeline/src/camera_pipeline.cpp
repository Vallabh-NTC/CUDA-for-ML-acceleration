#include <gst/gst.h>
#include <iostream>

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    const char* pipeline_desc =
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=I420 ! "
        "xvimagesink sync=false";

    GError* err = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeline_desc, &err);
    if (!pipeline) {
        std::cerr << "Failed to create pipeline: "
                  << (err ? err->message : "unknown error") << std::endl;
        if (err) g_error_free(err);
        return 1;
    }

    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Failed to set pipeline to PLAYING" << std::endl;
        gst_object_unref(pipeline);
        return 1;
    }

    // Wait until error or EOS
    GstBus* bus = gst_element_get_bus(pipeline);
    bool running = true;
    while (running) {
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus, GST_CLOCK_TIME_NONE,
            (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

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

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(pipeline);
    return 0;
}
