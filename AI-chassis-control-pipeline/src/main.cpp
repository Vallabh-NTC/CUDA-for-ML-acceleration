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
#include <optional>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include "UdpReceiver.hpp"
#include "AdmaDecoder.hpp"
#include "AdmaUdpReceiver.hpp"
#include "SARA_10.hpp"
#include "SARA_08.hpp"
#include "LWI01.hpp"
#include "Motor20.hpp"
#include "BrakeEV01.hpp"
#include "ESP21.hpp"
#include "ESP03.hpp"
#include "ESP05.hpp"
#include "Motor14.hpp"
#include "LHEPS03.hpp"
#include "KlimaSensor02.hpp"
#include "SARA_06.hpp"

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

struct ClusterSignalOffsets {
    int sara06;
    int sara10;
    int esp21;
    int esp03;
    int esp05;
    int lwi01;
    int lheps03;
    int klima_sensor_02;
    int motor20;
    int bremse_ev01;
    int motor14;
};

static const ClusterSignalOffsets kClusterSignalOffsets[65] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, 412, 542, 518, -1, 677, 597, -1, 629, 364, -1},
    {291, 274, -1, -1, -1, -1, -1, -1, -1, -1, 1098},
    {-1, 326, -1, -1, 1111, 56, 646, -1, 434, 261, -1},
    {294, 128, -1, -1, -1, -1, -1, 56, -1, -1, 120},
    {-1, 213, 375, 343, -1, 470, 1112, -1, 841, 48, -1},
    {98, 48, -1, -1, -1, -1, -1, -1, -1, -1, 65},
    {-1, 366, -1, -1, 48, 96, 290, -1, 233, 196, -1},
    {290, 1202, -1, -1, -1, -1, -1, -1, -1, -1, 1010},
    {-1, 620, 0, 577, -1, 172, 130, -1, 204, 536, -1},
    {890, 420, -1, -1, -1, -1, -1, -1, -1, -1, 833},
    {-1, 27, -1, -1, 1166, 1142, 1036, -1, 1158, 537, -1},
    {573, 436, -1, -1, -1, -1, -1, -1, -1, -1, 327},
    {-1, 671, 150, 1124, -1, 348, 1243, -1, 79, 958, -1},
    {48, 694, -1, -1, -1, -1, -1, -1, -1, -1, 40},
    {-1, 130, -1, -1, 665, 715, 45, -1, 612, 254, -1},
    {645, 1187, -1, -1, -1, -1, -1, -1, -1, -1, 83},
    {-1, 340, 202, 93, -1, 637, 16, -1, 349, 593, -1},
    {35, 749, -1, -1, -1, -1, -1, -1, -1, -1, 1147},
    {-1, 522, -1, -1, 161, 416, 649, -1, 359, 177, -1},
    {168, 214, -1, -1, -1, -1, -1, 488, -1, -1, 496},
    {-1, 792, 1179, 1203, -1, 242, 313, -1, 689, 1235, -1},
    {119, 207, -1, -1, -1, -1, -1, -1, -1, -1, 583},
    {-1, 552, -1, -1, 446, 802, 641, -1, 598, 284, -1},
    {319, 856, -1, -1, -1, -1, -1, -1, -1, -1, 652},
    {-1, 48, 341, 561, -1, 162, 40, -1, 78, 661, -1},
    {1194, 894, -1, -1, -1, -1, -1, -1, -1, -1, 617},
    {-1, 682, -1, -1, 254, 625, 1147, -1, 617, 392, -1},
    {116, 8, -1, -1, -1, -1, -1, -1, -1, -1, 41},
    {-1, 596, 74, 842, -1, 1027, 438, -1, 1003, 422, -1},
    {8, 457, -1, -1, -1, -1, -1, -1, -1, -1, 508},
    {-1, 367, -1, -1, 109, 32, 134, -1, 179, 16, -1},
    {45, 684, -1, -1, -1, -1, -1, -1, -1, -1, 781},
    {-1, 631, 196, 425, -1, 575, 188, -1, 559, 172, -1},
    {148, 348, -1, -1, -1, -1, -1, -1, -1, -1, 558},
    {-1, 457, -1, -1, 848, 192, 157, -1, 65, 116, -1},
    {398, 306, -1, -1, -1, -1, -1, 189, -1, -1, 515},
    {-1, 222, 1106, 312, -1, 1061, 842, -1, 1098, 544, -1},
    {395, 111, -1, -1, -1, -1, -1, -1, -1, -1, 573},
    {-1, 398, -1, -1, 468, 357, 729, -1, 349, 684, -1},
    {460, 747, -1, -1, -1, -1, -1, -1, -1, -1, 1200},
    {-1, 0, 340, 409, -1, 653, 553, -1, 497, 372, -1},
    {43, 1108, -1, -1, -1, -1, -1, -1, -1, -1, 719},
    {-1, 402, -1, -1, 692, 151, 1141, -1, 8, 454, -1},
    {21, 242, -1, -1, -1, -1, -1, -1, -1, -1, 427},
    {-1, 324, 628, 219, -1, 678, 45, -1, 964, 248, -1},
    {703, 156, -1, -1, -1, -1, -1, -1, -1, -1, 674},
    {-1, 424, -1, -1, 533, 305, 717, -1, 117, 582, -1},
    {497, 880, -1, -1, -1, -1, -1, -1, -1, -1, 100},
    {-1, 96, 270, 24, -1, 185, 8, -1, 374, 48, -1},
    {448, 84, -1, -1, -1, -1, -1, -1, -1, -1, 137},
    {-1, 1057, -1, -1, 1128, 511, 1163, -1, 68, 1029, -1},
    {229, 88, -1, -1, -1, -1, -1, 460, -1, -1, 54},
    {-1, 683, 659, 606, -1, 389, 222, -1, 365, 381, -1},
    {165, 261, -1, -1, -1, -1, -1, -1, -1, -1, 608},
    {-1, 112, -1, -1, 604, 83, 472, -1, 620, 728, -1},
    {1176, 497, -1, -1, -1, -1, -1, -1, -1, -1, 820},
    {-1, 660, 434, 402, -1, 221, 335, -1, 213, 40, -1},
    {1200, 1075, -1, -1, -1, -1, -1, -1, -1, -1, 1192},
    {-1, 943, -1, -1, 758, 418, 524, -1, 446, 894, -1},
    {132, 189, -1, -1, -1, -1, -1, -1, -1, -1, 68},
    {-1, 16, 480, 562, -1, 818, 1080, -1, 1064, 728, -1},
    {210, 568, -1, -1, -1, -1, -1, -1, -1, -1, 36},
    {-1, 16, -1, -1, 41, 250, 137, -1, 170, 57, -1},
    {152, 341, -1, -1, -1, -1, -1, -1, -1, -1, 208}
};

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
            "ax,ay,az,ox,oy,oz,steer,steer_spd,gas,brake,v,"
            "esp_v_signal,sara06_accel_x,sara10_accel_x,sara06_accel_y,sara10_accel_y,sara06_omega_z,sara10_omega_z,"
            "lwi_angle,lwi_angle_sign,lwi_speed,lwi_speed_sign,eps_torque,eps_torque_sign,external_temp,"
            "gas_pedal_pos,brake_driver,brake_pedal_pos,esp_brake_pressure,motor14_mo_bls,"
            "wheel_fl,wheel_fr,wheel_rl,wheel_rr,esp_eingriff,"
            "admaAccX,admaAccY,admaAccZ,admaYaw,admaPitch,admaRoll,admaKmh,image\n";
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

    // -------- ADMA receiver (v3.3.4) --------
    adma::AdmaPacketDecoder adma_decoder(adma::ProtocolVersion::V334);
    adma::AdmaUdpReceiver adma_receiver(
        "195.0.5.4",
        static_cast<uint16_t>(1021),
        std::optional<std::string>{"195.0.5.50"});

    float ax=NAN, ay=NAN, az=NAN;
    float ox=NAN, oy=NAN, oz=NAN;
    float steer=NAN, steer_spd=NAN;
    float gas=NAN, brake=NAN, v=NAN;

    float sara06_ax = NAN, sara06_ay = NAN, sara06_oz = NAN;
    float sara10_ax = NAN, sara10_ay = NAN, sara10_oz = NAN;
    float esp_v_signal = NAN;
    uint8_t esp_eingriff = 0;

    float lwi_angle = NAN;
    uint8_t lwi_angle_sign = 0;
    float lwi_speed = NAN;
    uint8_t lwi_speed_sign = 0;

    float eps_torque = NAN;
    uint8_t eps_torque_sign = 0;
    float external_temp = NAN;

    float gas_pedal_pos = NAN;
    uint8_t brake_driver = 0;
    float brake_pedal_pos = NAN;
    float esp_brake_pressure = NAN;
    uint8_t motor14_mo_bls = 0;

    float wheel_fl = NAN, wheel_fr = NAN, wheel_rl = NAN, wheel_rr = NAN;

    double adma_acc_x = NAN;
    double adma_acc_y = NAN;
    double adma_acc_z = NAN;
    double adma_yaw = NAN;
    double adma_pitch = NAN;
    double adma_roll = NAN;
    double adma_kmh = NAN;

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

        sara06_ax = NAN; sara06_ay = NAN; sara06_oz = NAN;
        sara10_ax = NAN; sara10_ay = NAN; sara10_oz = NAN;
        esp_v_signal = NAN; esp_eingriff = 0;
        lwi_angle = NAN; lwi_angle_sign = 0; lwi_speed = NAN; lwi_speed_sign = 0;
        eps_torque = NAN; eps_torque_sign = 0;
        external_temp = NAN;
        gas_pedal_pos = NAN;
        brake_driver = 0;
        brake_pedal_pos = NAN;
        esp_brake_pressure = NAN;
        motor14_mo_bls = 0;
        wheel_fl = NAN; wheel_fr = NAN; wheel_rl = NAN; wheel_rr = NAN;

        if (cluster >= 1 && cluster <= 64) {
            const ClusterSignalOffsets& off = kClusterSignalOffsets[cluster];

            if (off.sara06 >= 0) {
                SARA_06 s06;
                s06.decode(pdus + off.sara06);
                sara06_ax = s06.data().accel_x;
                sara06_ay = s06.data().accel_y;
                sara06_oz = s06.data().omega_z;
            }

            if (off.sara10 >= 0) {
                SARA_10 s10;
                s10.decode(pdus + off.sara10);
                sara10_ax = s10.data().accel_x;
                sara10_ay = s10.data().accel_y;
                sara10_oz = s10.data().omega_z;
            }

            if (off.esp21 >= 0) {
                ESP21 e;
                e.decode(pdus + off.esp21);
                esp_v_signal = e.data().vehicle_speed;
                esp_eingriff = e.data().esp_intervention;
                v = esp_v_signal;
            }

            if (off.esp03 >= 0) {
                ESP03 e3;
                e3.decode(pdus + off.esp03);
                wheel_fl = e3.data().wheel_speed_fl;
                wheel_fr = e3.data().wheel_speed_fr;
                wheel_rl = e3.data().wheel_speed_rl;
                wheel_rr = e3.data().wheel_speed_rr;
            }

            if (off.esp05 >= 0) {
                ESP05 e5;
                e5.decode(pdus + off.esp05);
                esp_brake_pressure = e5.data().brake_pressure;
            }

            if (off.lwi01 >= 0) {
                LWI01 lwi;
                lwi.decode(pdus + off.lwi01);
                lwi_angle = lwi.angle;
                lwi_angle_sign = lwi.angle_sign;
                lwi_speed = lwi.speed;
                lwi_speed_sign = lwi.speed_sign;
                steer = lwi_angle;
                steer_spd = lwi_speed;
            }

            if (off.lheps03 >= 0) {
                LHEPS03 eps;
                eps.decode(pdus + off.lheps03);
                eps_torque = eps.data().steering_torque;
                eps_torque_sign = eps.data().steering_torque_sign;
            }

            if (off.klima_sensor_02 >= 0) {
                KlimaSensor02 k;
                k.decode(pdus + off.klima_sensor_02);
                external_temp = k.data().external_temperature;
            }

            if (off.motor20 >= 0) {
                Motor20 m20;
                m20.decode(pdus + off.motor20);
                gas_pedal_pos = m20.data().gas_percent;
                gas = gas_pedal_pos;
            }

            if (off.bremse_ev01 >= 0) {
                BrakeEV01 br;
                br.decode(pdus + off.bremse_ev01);
                brake_driver = br.driver_brakes;
                brake_pedal_pos = br.pedal_position;
                brake = brake_pedal_pos;
            }

            if (off.motor14 >= 0) {
                Motor14 m14;
                m14.decode(pdus + off.motor14);
                motor14_mo_bls = m14.data().mo_bls;
            }
        }

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

        try {
            const auto adma_payload = adma_receiver.receive();
            const auto decoded = adma_decoder.decode(adma_payload);

            adma_acc_x = decoded.kinematics.accx_g;
            adma_acc_y = decoded.kinematics.accy_g;
            adma_acc_z = decoded.kinematics.accz_g;
            adma_kmh = decoded.kinematics.speed_kmh;

            if (decoded.v334.has_value()) {
                const auto& adma_packet = decoded.v334.value();
                adma_roll = static_cast<double>(adma_packet.insroll) * 0.01;
                adma_pitch = static_cast<double>(adma_packet.inspitch) * 0.01;
                adma_yaw = static_cast<double>(adma_packet.insyaw) * 0.01;
            }
        } catch (const std::exception& ex) {
            std::cerr << "WARN: ADMA receive/decode failed: " << ex.what() << "\n";
            adma_acc_x = NAN;
            adma_acc_y = NAN;
            adma_acc_z = NAN;
            adma_yaw = NAN;
            adma_pitch = NAN;
            adma_roll = NAN;
            adma_kmh = NAN;
        }

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
            << esp_v_signal << ","
            << sara06_ax << "," << sara10_ax << ","
            << sara06_ay << "," << sara10_ay << ","
            << sara06_oz << "," << sara10_oz << ","
            << lwi_angle << "," << static_cast<unsigned>(lwi_angle_sign) << ","
            << lwi_speed << "," << static_cast<unsigned>(lwi_speed_sign) << ","
            << eps_torque << "," << static_cast<unsigned>(eps_torque_sign) << ","
            << external_temp << ","
            << gas_pedal_pos << "," << static_cast<unsigned>(brake_driver) << ","
            << brake_pedal_pos << "," << esp_brake_pressure << ","
            << static_cast<unsigned>(motor14_mo_bls) << ","
            << wheel_fl << "," << wheel_fr << "," << wheel_rl << "," << wheel_rr << ","
            << static_cast<unsigned>(esp_eingriff) << ","
            << adma_acc_x << "," << adma_acc_y << "," << adma_acc_z << ","
            << adma_yaw << "," << adma_pitch << "," << adma_roll << "," << adma_kmh << ","
            << img_path << "\n";
        csv.flush();
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(sink_elem);
    gst_object_unref(pipeline);
    return 0;
}
