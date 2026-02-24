// udp_decoder_gst.cpp
// UDP -> decode -> grab ONE USB-camera frame via GStreamer -> save idx.jpg -> log CSV row with same idx
// No threads, no timestamp matching. Pure 1:1 pairing by counter.
//
// Build (example):
//   g++ -O2 -std=c++17 udp_decoder_gst.cpp -o udp_decoder_gst `pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0`
//
// Run examples:
//   ./udp_decoder_gst --device /dev/video0 --w 2560 --h 720 --fps 30 --crop right --crop-px 1280
//   ./udp_decoder_gst --device /dev/video0 --w 1280 --h 720 --fps 60 --crop right --crop-px 640
//   ./udp_decoder_gst --device /dev/video0 --w 672  --h 376 --fps 100 --crop none
//
// Notes:
// - Uses YUY2 (matches your working gst-launch).
// - Uses nvvidconv + NVMM/NV12 + nvjpegenc by default; use --no-nv to fall back to CPU jpegenc.
// - If the camera doesn't support requested caps, you'll get not-negotiated.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include "UdpReceiver.hpp"
#include "SARA_10.hpp"
#include "SARA_08.hpp"
#include "LWI01.hpp"
#include "Motor20.hpp"
#include "BrakeEV01.hpp"
#include "ESP21.hpp"

static inline void print_line(
    int cluster,
    float ax, float ay, float az,
    float ox, float oy, float oz,
    float steer, float steer_spd,
    float gas, float brake, float v,
    const timespec& ts)
{
    std::tm tm_local{};
    localtime_r(&ts.tv_sec, &tm_local);

    char tb[32];
    std::strftime(tb, sizeof(tb), "%H:%M:%S", &tm_local);
    long ms = ts.tv_nsec / 1000000;

    std::cout << std::fixed << std::setprecision(3)
              << "[" << tb << "." << std::setw(3) << std::setfill('0') << ms
              << std::setfill(' ') << "] "
              << "C" << cluster
              << " | ax=" << ax << " ay=" << ay << " az=" << az
              << " | ox=" << ox << " oy=" << oy << " oz=" << oz
              << " | steer=" << steer << " spd=" << steer_spd
              << " | gas=" << gas << " brake=" << brake
              << " v=" << v
              << "\n";
}

static bool save_one_jpeg_from_appsink(GstAppSink* appsink, const char* filepath)
{
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return false;

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return false;
    }

    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return false;
    }

    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return false;
    }

    out.write(reinterpret_cast<const char*>(map.data),
              static_cast<std::streamsize>(map.size));
    out.close();

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    return true;
}

static bool check_bus_nonblocking(GstElement* pipeline)
{
    GstBus* bus = gst_element_get_bus(pipeline);
    if (!bus) return true;

    bool ok = true;
    while (true) {
        GstMessage* msg = gst_bus_pop(bus);
        if (!msg) break;

        switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gchar* dbg = nullptr;
            gst_message_parse_error(msg, &err, &dbg);
            std::cerr << "GStreamer ERROR: " << (err ? err->message : "unknown") << "\n";
            if (dbg) std::cerr << "  Debug: " << dbg << "\n";
            if (err) g_error_free(err);
            if (dbg) g_free(dbg);
            ok = false;
            break;
        }
        case GST_MESSAGE_EOS:
            std::cerr << "GStreamer EOS\n";
            ok = false;
            break;
        default:
            break;
        }
        gst_message_unref(msg);
        if (!ok) break;
    }

    gst_object_unref(bus);
    return ok;
}

struct Opts {
    std::string device = "/dev/video0";
    int width = 2560;
    int height = 720;
    int fps = 30;

    // "none" | "left" | "right"
    std::string crop_mode = "none";
    int crop_px = 0;

    int jpeg_quality = 85;
    bool use_nv = true;
};

static void usage(const char* argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " [--device /dev/video0] [--w W] [--h H] --fps N\n"
        << "        [--crop none|left|right] [--crop-px N]\n"
        << "        [--jpegq 1..100] [--no-nv]\n\n"
        << "Examples:\n"
        << "  " << argv0 << " --device /dev/video0 --w 2560 --h 720 --fps 30 --crop right --crop-px 1280\n"
        << "  " << argv0 << " --device /dev/video0 --w 1280 --h 720 --fps 60 --crop right --crop-px 640\n"
        << "  " << argv0 << " --device /dev/video0 --w 672  --h 376 --fps 100 --crop none\n";
}

static bool parse_args(int argc, char** argv, Opts& o)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "--device") {
            const char* v = need("--device"); if (!v) return false;
            o.device = v;
        } else if (a == "--w") {
            const char* v = need("--w"); if (!v) return false;
            o.width = std::atoi(v);
        } else if (a == "--h") {
            const char* v = need("--h"); if (!v) return false;
            o.height = std::atoi(v);
        } else if (a == "--fps") {
            const char* v = need("--fps"); if (!v) return false;
            o.fps = std::atoi(v);
        } else if (a == "--crop") {
            const char* v = need("--crop"); if (!v) return false;
            o.crop_mode = v;
        } else if (a == "--crop-px") {
            const char* v = need("--crop-px"); if (!v) return false;
            o.crop_px = std::atoi(v);
        } else if (a == "--jpegq") {
            const char* v = need("--jpegq"); if (!v) return false;
            o.jpeg_quality = std::atoi(v);
        } else if (a == "--no-nv") {
            o.use_nv = false;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown arg: " << a << "\n";
            usage(argv[0]);
            return false;
        }
    }

    if (o.fps <= 0) { std::cerr << "Invalid --fps\n"; return false; }
    if (o.width <= 0 || o.height <= 0) { std::cerr << "Invalid --w/--h\n"; return false; }

    if (!(o.crop_mode == "none" || o.crop_mode == "left" || o.crop_mode == "right")) {
        std::cerr << "Invalid --crop (use none|left|right)\n";
        return false;
    }
    if (o.crop_mode != "none") {
        if (o.crop_px <= 0 || o.crop_px >= o.width) {
            std::cerr << "--crop-px must be in [1, width-1]\n";
            return false;
        }
    } else {
        o.crop_px = 0;
    }

    if (o.jpeg_quality < 1) o.jpeg_quality = 1;
    if (o.jpeg_quality > 100) o.jpeg_quality = 100;

    return true;
}

static std::string build_pipeline(const Opts& o)
{
    // Matches your working gst-launch pipeline style, but ends in nvjpegenc + appsink.
    const int out_w = o.width - o.crop_px;
    const int out_h = o.height;

    std::ostringstream ss;

    ss << "v4l2src device=" << o.device << " io-mode=2 ! "
       << "video/x-raw,format=YUY2,width=" << o.width << ",height=" << o.height
       << ",framerate=" << o.fps << "/1 ! "
       << "videoconvert ! ";

    if (o.crop_mode == "right") ss << "videocrop right=" << o.crop_px << " ! ";
    else if (o.crop_mode == "left") ss << "videocrop left=" << o.crop_px << " ! ";

    if (o.use_nv) {
        // Don't force framerate after nvvidconv; it can sometimes break negotiation.
        ss << "nvvidconv ! "
           << "video/x-raw(memory:NVMM),format=NV12,width=" << out_w << ",height=" << out_h << " ! "
           << "nvjpegenc quality=" << o.jpeg_quality << " ! ";
    } else {
        ss << "videoconvert ! video/x-raw,format=I420 ! "
           << "jpegenc quality=" << o.jpeg_quality << " ! ";
    }

    ss << "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true";

    return ss.str();
}

int main(int argc, char** argv)
{
    // -------- Output folders --------
    std::filesystem::create_directories("log/images");

    // -------- CSV log --------
    std::ofstream csv("log/telemetry.csv");
    if (!csv) {
        std::cerr << "ERROR: cannot open log/telemetry.csv\n";
        return 1;
    }
    csv << "idx,unix_sec,unix_nsec,cluster,"
           "ax,ay,az,ox,oy,oz,steer,steer_spd,gas,brake,v,image\n";
    csv.flush();

    // -------- Parse args --------
    Opts opts{};
    if (!parse_args(argc, argv, opts)) return 1;

    // -------- GStreamer init --------
    gst_init(&argc, &argv);

    // -------- Pipeline --------
    const std::string pipeline_desc = build_pipeline(opts);
    std::cerr << "pipeline:\n  " << pipeline_desc << "\n";

    GError* err = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeline_desc.c_str(), &err);
    if (!pipeline) {
        std::cerr << "ERROR: gst_parse_launch failed\n";
        if (err) { std::cerr << "  " << err->message << "\n"; g_error_free(err); }
        return 1;
    }
    if (err) {
        std::cerr << "GStreamer parse warning/error: " << err->message << "\n";
        g_error_free(err);
        err = nullptr;
    }

    GstElement* sink_elem = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!sink_elem) {
        std::cerr << "ERROR: cannot find appsink element 'sink'\n";
        gst_object_unref(pipeline);
        return 1;
    }
    GstAppSink* appsink = GST_APP_SINK(sink_elem);

    // Start pipeline
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "ERROR: failed to set GStreamer pipeline to PLAYING\n";
        gst_object_unref(sink_elem);
        gst_object_unref(pipeline);
        return 1;
    }

    // -------- UDP receiver --------
    UdpReceiver receiver(1500);
    unsigned char buf[65536];

    float ax=NAN, ay=NAN, az=NAN;
    float ox=NAN, oy=NAN, oz=NAN;
    float steer=NAN, steer_spd=NAN;
    float gas=NAN, brake=NAN, v=NAN;

    uint64_t idx = 0;

    while (true) {
        if (!check_bus_nonblocking(pipeline)) {
            std::cerr << "Stopping due to GStreamer error/EOS.\n";
            break;
        }

        int n = receiver.receive(buf, sizeof(buf));
        if (n < 3) continue;

        int cluster = buf[0];
        const unsigned char* pdus = buf + 3;

        timespec ts{};
        clock_gettime(CLOCK_REALTIME, &ts);

        bool has = false;

        switch (cluster) {

        case 1: {
            BrakeEV01 br; br.decode(pdus + 364); brake = br.brake_percent;
            SARA_10 s10; s10.decode(pdus + 412);
            SARA_08 s08; s08.decode(pdus + 437);
            ESP21 e; e.decode(pdus + 542); v = e.data().vehicle_speed;
            Motor20 m; m.decode(pdus + 629); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 677); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 2: {
            SARA_10 s10; s10.decode(pdus + 274);
            SARA_08 s08; s08.decode(pdus + 348);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 3: {
            LWI01 l; l.decode(pdus + 56); steer = l.angle; steer_spd = l.speed;
            BrakeEV01 br; br.decode(pdus + 261); brake = br.brake_percent;
            SARA_10 s10; s10.decode(pdus + 326);
            SARA_08 s08; s08.decode(pdus + 425);
            Motor20 m; m.decode(pdus + 434); gas = m.data().gas_percent;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 4: {
            SARA_10 s10; s10.decode(pdus + 128);
            SARA_08 s08; s08.decode(pdus + 385);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 5: {
            BrakeEV01 br; br.decode(pdus + 48); brake = br.brake_percent;
            SARA_10 s10; s10.decode(pdus + 213);
            SARA_08 s08; s08.decode(pdus + 230);
            ESP21 e; e.decode(pdus + 375); v = e.data().vehicle_speed;
            LWI01 l; l.decode(pdus + 470); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 6: {
            SARA_10 s10; s10.decode(pdus + 48);
            SARA_08 s08; s08.decode(pdus + 73);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 7: {
            LWI01 l; l.decode(pdus + 96); steer = l.angle; steer_spd = l.speed;
            BrakeEV01 br; br.decode(pdus + 196); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 233); gas = m.data().gas_percent;
            SARA_08 s08; s08.decode(pdus + 241);
            SARA_10 s10; s10.decode(pdus + 366);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 8: {
            SARA_08 s08; s08.decode(pdus + 349);
            SARA_10 s10; s10.decode(pdus + 1202);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 9: {
            ESP21 e; e.decode(pdus + 0); v = e.data().vehicle_speed;
            Motor20 m; m.decode(pdus + 204); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 172); steer = l.angle; steer_spd = l.speed;
            BrakeEV01 br; br.decode(pdus + 536); brake = br.brake_percent;
            SARA_08 s08; s08.decode(pdus + 560);
            SARA_10 s10; s10.decode(pdus + 620);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 10: {
            SARA_10 s10; s10.decode(pdus + 420);
            SARA_08 s08; s08.decode(pdus + 849);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 11: {
            SARA_10 s10; s10.decode(pdus + 27);
            BrakeEV01 br; br.decode(pdus + 537); brake = br.brake_percent;
            SARA_08 s08; s08.decode(pdus + 561);
            LWI01 l; l.decode(pdus + 1142); steer = l.angle; steer_spd = l.speed;
            Motor20 m; m.decode(pdus + 1158); gas = m.data().gas_percent;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 12: {
            SARA_08 s08; s08.decode(pdus + 403);
            SARA_10 s10; s10.decode(pdus + 436);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 13: {
            Motor20 m; m.decode(pdus + 79); gas = m.data().gas_percent;
            ESP21 e; e.decode(pdus + 150); v = e.data().vehicle_speed;
            LWI01 l; l.decode(pdus + 348); steer = l.angle; steer_spd = l.speed;
            SARA_10 s10; s10.decode(pdus + 671);
            SARA_08 s08; s08.decode(pdus + 820);
            BrakeEV01 br; br.decode(pdus + 958); brake = br.brake_percent;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 14: {
            SARA_08 s08; s08.decode(pdus + 121);
            az = s08.data().accel_z;
            ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 15: {
            SARA_10 s10; s10.decode(pdus + 130);
            SARA_08 s08; s08.decode(pdus + 192);
            BrakeEV01 br; br.decode(pdus + 254); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 612); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 715); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 16: {
            SARA_08 s08; s08.decode(pdus + 866);
            SARA_10 s10; s10.decode(pdus + 1187);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 17: {
            SARA_08 s08; s08.decode(pdus + 125);
            SARA_10 s10; s10.decode(pdus + 340);
            Motor20 m; m.decode(pdus + 349); gas = m.data().gas_percent;
            BrakeEV01 br; br.decode(pdus + 593); brake = br.brake_percent;
            LWI01 l; l.decode(pdus + 637); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 18: {
            SARA_10 s10; s10.decode(pdus + 749);
            SARA_08 s08; s08.decode(pdus + 1114);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 19: {
            BrakeEV01 br; br.decode(pdus + 177); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 359); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 416); steer = l.angle; steer_spd = l.speed;
            SARA_10 s10; s10.decode(pdus + 522);
            SARA_08 s08; s08.decode(pdus + 813);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 20: {
            SARA_10 s10; s10.decode(pdus + 214);
            SARA_08 s08; s08.decode(pdus + 260);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 21: {
            SARA_08 s08; s08.decode(pdus + 45);
            SARA_10 s10; s10.decode(pdus + 792);
            Motor20 m; m.decode(pdus + 689); gas = m.data().gas_percent;
            BrakeEV01 br; br.decode(pdus + 1235); brake = br.brake_percent;
            LWI01 l; l.decode(pdus + 242); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 22: {
            SARA_10 s10; s10.decode(pdus + 207);
            SARA_08 s08; s08.decode(pdus + 374);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 23: {
            SARA_08 s08; s08.decode(pdus + 206);
            BrakeEV01 br; br.decode(pdus + 284); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 598); gas = m.data().gas_percent;
            SARA_10 s10; s10.decode(pdus + 552);
            LWI01 l; l.decode(pdus + 802); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 24: {
            SARA_08 s08; s08.decode(pdus + 77);
            SARA_10 s10; s10.decode(pdus + 856);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 25: {
            SARA_10 s10; s10.decode(pdus + 48);
            SARA_08 s08; s08.decode(pdus + 86);
            Motor20 m; m.decode(pdus + 78); gas = m.data().gas_percent;
            BrakeEV01 br; br.decode(pdus + 661); brake = br.brake_percent;
            LWI01 l; l.decode(pdus + 162); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 26: {
            SARA_08 s08; s08.decode(pdus + 273);
            SARA_10 s10; s10.decode(pdus + 894);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 27: {
            BrakeEV01 br; br.decode(pdus + 392); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 617); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 625); steer = l.angle; steer_spd = l.speed;
            SARA_10 s10; s10.decode(pdus + 682);
            SARA_08 s08; s08.decode(pdus + 1130);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 28: {
            SARA_10 s10; s10.decode(pdus + 8);
            SARA_08 s08; s08.decode(pdus + 70);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 29: {
            SARA_08 s08; s08.decode(pdus + 206);
            BrakeEV01 br; br.decode(pdus + 422); brake = br.brake_percent;
            SARA_10 s10; s10.decode(pdus + 596);
            Motor20 m; m.decode(pdus + 1003); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 1027); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 30: {
            SARA_08 s08; s08.decode(pdus + 288);
            SARA_10 s10; s10.decode(pdus + 457);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 31: {
            BrakeEV01 br; br.decode(pdus + 16); brake = br.brake_percent;
            LWI01 l; l.decode(pdus + 32); steer = l.angle; steer_spd = l.speed;
            Motor20 m; m.decode(pdus + 179); gas = m.data().gas_percent;
            SARA_10 s10; s10.decode(pdus + 367);
            SARA_08 s08; s08.decode(pdus + 125);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 32: {
            SARA_08 s08; s08.decode(pdus + 654);
            SARA_10 s10; s10.decode(pdus + 684);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 33: {
            BrakeEV01 br; br.decode(pdus + 172); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 559); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 575); steer = l.angle; steer_spd = l.speed;
            SARA_08 s08; s08.decode(pdus + 518);
            SARA_10 s10; s10.decode(pdus + 631);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 34: {
            SARA_10 s10; s10.decode(pdus + 348);
            SARA_08 s08; s08.decode(pdus + 661);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 35: {
            BrakeEV01 br; br.decode(pdus + 116); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 65); gas = m.data().gas_percent;
            SARA_10 s10; s10.decode(pdus + 457);
            SARA_08 s08; s08.decode(pdus + 614);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 36: {
            SARA_08 s08; s08.decode(pdus + 32);
            SARA_10 s10; s10.decode(pdus + 306);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 37: {
            SARA_10 s10; s10.decode(pdus + 222);
            SARA_08 s08; s08.decode(pdus + 252);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 38: {
            SARA_10 s10; s10.decode(pdus + 111);
            SARA_08 s08; s08.decode(pdus + 759);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 39: {
            SARA_10 s10; s10.decode(pdus + 398);
            SARA_08 s08; s08.decode(pdus + 389);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 40: {
            SARA_10 s10; s10.decode(pdus + 747);
            SARA_08 s08; s08.decode(pdus + 1062);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 41: {
            SARA_10 s10; s10.decode(pdus + 0);
            SARA_08 s08; s08.decode(pdus + 25);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 42: {
            SARA_10 s10; s10.decode(pdus + 1108);
            SARA_08 s08; s08.decode(pdus + 108);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 43: {
            SARA_10 s10; s10.decode(pdus + 402);
            SARA_08 s08; s08.decode(pdus + 261);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 44: {
            SARA_10 s10; s10.decode(pdus + 242);
            SARA_08 s08; s08.decode(pdus + 378);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 45: {
            SARA_10 s10; s10.decode(pdus + 324);
            SARA_08 s08; s08.decode(pdus + 855);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 46: {
            SARA_10 s10; s10.decode(pdus + 156);
            SARA_08 s08; s08.decode(pdus + 418);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 47: {
            SARA_10 s10; s10.decode(pdus + 424);
            SARA_08 s08; s08.decode(pdus + 0);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 48: {
            SARA_10 s10; s10.decode(pdus + 880);
            SARA_08 s08; s08.decode(pdus + 158);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 49: {
            SARA_10 s10; s10.decode(pdus + 96);
            SARA_08 s08; s08.decode(pdus + 553);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 50: {
            SARA_10 s10; s10.decode(pdus + 84);
            SARA_08 s08; s08.decode(pdus + 153);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 51: {
            SARA_10 s10; s10.decode(pdus + 1057);
            SARA_08 s08; s08.decode(pdus + 464);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 52: {
            SARA_10 s10; s10.decode(pdus + 88);
            SARA_08 s08; s08.decode(pdus + 150);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 53: {
            SARA_10 s10; s10.decode(pdus + 683);
            SARA_08 s08; s08.decode(pdus + 230);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 54: {
            SARA_10 s10; s10.decode(pdus + 261);
            SARA_08 s08; s08.decode(pdus + 656);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 55: {
            SARA_10 s10; s10.decode(pdus + 112);
            SARA_08 s08; s08.decode(pdus + 0);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 56: {
            SARA_10 s10; s10.decode(pdus + 497);
            SARA_08 s08; s08.decode(pdus + 680);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 57: {
            SARA_10 s10; s10.decode(pdus + 660);
            SARA_08 s08; s08.decode(pdus + 603);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 58: {
            SARA_10 s10; s10.decode(pdus + 1075);
            SARA_08 s08; s08.decode(pdus + 838);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 59: {
            SARA_10 s10; s10.decode(pdus + 943);
            SARA_08 s08; s08.decode(pdus + 369);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 60: {
            SARA_10 s10; s10.decode(pdus + 189);
            SARA_08 s08; s08.decode(pdus + 307);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 61: {
            SARA_10 s10; s10.decode(pdus + 16);
            SARA_08 s08; s08.decode(pdus + 110);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 62: {
            SARA_10 s10; s10.decode(pdus + 568);
            SARA_08 s08; s08.decode(pdus + 698);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 63: {
            SARA_10 s10; s10.decode(pdus + 16);
            BrakeEV01 br; br.decode(pdus + 57); brake = br.brake_percent;
            SARA_08 s08; s08.decode(pdus + 161);
            Motor20 m; m.decode(pdus + 170); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 250); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        case 64: {
            SARA_10 s10; s10.decode(pdus + 341);
            SARA_08 s08; s08.decode(pdus + 468);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        default:
            break;
        }

        if (!has) continue;

        ++idx;

        char img_path[256];
        std::snprintf(img_path, sizeof(img_path), "log/images/%06llu.jpg",
                      (unsigned long long)idx);

        if (!save_one_jpeg_from_appsink(appsink, img_path)) {
            std::cerr << "WARN: failed to grab/save jpeg at idx=" << idx << "\n";
            continue;
        }

        print_line(cluster,
                   ax, ay, az,
                   ox, oy, oz,
                   steer, steer_spd,
                   gas, brake, v,
                   ts);

        csv << idx << ","
            << ts.tv_sec << "," << ts.tv_nsec << ","
            << cluster << ","
            << ax << "," << ay << "," << az << ","
            << ox << "," << oy << "," << oz << ","
            << steer << "," << steer_spd << ","
            << gas << "," << brake << "," << v << ","
            << img_path << "\n";
        csv.flush();
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(sink_elem);
    gst_object_unref(pipeline);
    return 0;
}
