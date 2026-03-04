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
#include <thread>
#include <mutex>
#include <condition_variable>

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
#include "Getriebe11.hpp"
#include "LHEPS03.hpp"
#include "KlimaSensor02.hpp"
#include "SARA_06.hpp"

static inline void print_line(
    int adma_kf_status,
    int adma_kf_lat_stimulated,
    int adma_kf_long_stimulated,
    int adma_kf_steady_state,
    double adma_ins_vel_hor_x,
    double adma_ins_vel_hor_y,
    double adma_acc_body_y,
    float flex_lwi_01_lwi_lenkradwinkel,
    const timespec& ts)
{
    std::tm tm_local{};
    localtime_r(&ts.tv_sec, &tm_local);

    char tb[32];
    std::strftime(tb, sizeof(tb), "%H:%M:%S", &tm_local);
    long ms = ts.tv_nsec / 1000000;

    const double vel_res = std::sqrt(
        adma_ins_vel_hor_x * adma_ins_vel_hor_x +
        adma_ins_vel_hor_y * adma_ins_vel_hor_y);

    std::cout << std::fixed << std::setprecision(3)
              << "[" << tb << "." << std::setw(3) << std::setfill('0') << ms
              << std::setfill(' ') << "] "
              << "\n | kf_status=" << adma_kf_status
              << "\n | kf_lat_stim=" << adma_kf_lat_stimulated
              << "\n | kf_long_stim=" << adma_kf_long_stimulated
              << "\n | kf_steady=" << adma_kf_steady_state
              << "\n | vel_xy_res=" << vel_res
              << "\n | acc_body_y=" << adma_acc_body_y
              << "\n | lenkradwinkel=" << flex_lwi_01_lwi_lenkradwinkel
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

static bool save_jpeg_from_sample(GstSample* sample, const char* filepath)
{
    if (!sample) return false;

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) return false;

    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) return false;

    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        gst_buffer_unmap(buffer, &map);
        return false;
    }

    out.write(reinterpret_cast<const char*>(map.data),
              static_cast<std::streamsize>(map.size));
    out.close();

    gst_buffer_unmap(buffer, &map);
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

static const int kGetriebe11Offsets[65] = {
    -1,
    -1, 1179, -1, 24, -1, 513, -1, 504, -1, 573, -1, 276, -1, 307, -1, 91,
    -1, 512, -1, 331, -1, 415, -1, 213, -1, 1203, -1, 424, -1, 604, -1, 455,
    -1, 1043, -1, 431, -1, 342, -1, 0, -1, 1040, -1, 310, -1, 181, -1, 548,
    -1, 807, -1, 365, -1, 640, -1, 399, -1, 401, -1, 76, -1, 92, -1, 0
};

static inline timespec now_realtime()
{
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts;
}

static std::filesystem::path create_run_log_dir()
{
    namespace fs = std::filesystem;

    const fs::path base_log_dir("log");
    fs::create_directories(base_log_dir);

    int next_log_index = 1;
    for (const auto& entry : fs::directory_iterator(base_log_dir)) {
        if (!entry.is_directory()) continue;

        const std::string name = entry.path().filename().string();
        const std::size_t marker_pos = name.rfind("_log_");
        if (marker_pos == std::string::npos) continue;

        const std::string index_str = name.substr(marker_pos + 5);
        if (index_str.empty()) continue;

        bool all_digits = true;
        for (const char c : index_str) {
            if (c < '0' || c > '9') {
                all_digits = false;
                break;
            }
        }
        if (!all_digits) continue;

        const int existing_index = std::stoi(index_str);
        if (existing_index >= next_log_index) {
            next_log_index = existing_index + 1;
        }
    }

    const timespec ts = now_realtime();
    std::tm tm_local{};
    localtime_r(&ts.tv_sec, &tm_local);

    char time_buf[32];
    std::strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M", &tm_local);

    std::ostringstream run_name;
    run_name << time_buf << "_log_" << next_log_index;

    const fs::path run_dir = base_log_dir / run_name.str();
    fs::create_directories(run_dir / "images");
    return run_dir;
}

static inline float qnanf()
{
    return 0.0f;
}

static inline double qnand()
{
    return 0.0;
}

struct FlexSnapshot {
    timespec ts{};
    int cluster = 0;

    float flex_SARA_06_SARA_Accel_X_010 = qnanf(), flex_SARA_06_SARA_Accel_Y_010 = qnanf(), flex_SARA_06_SARA_Omega_Z_010 = qnanf();
    float flex_SARA_10_SARA_Accel_X_b = qnanf(), flex_SARA_10_SARA_Accel_Y_b = qnanf(), flex_SARA_10_SARA_Omega_Z_b = qnanf();
    float flex_ESP_21_ESP_v_Signal = qnanf();
    uint8_t flex_ESP_21_ESP_Eingriff = 0;

    float flex_LWI_01_LWI_Lenkradwinkel = qnanf();
    uint8_t flex_LWI_01_LWI_VZ_Lenkradwinkel = 0;
    float flex_LWI_01_LWI_Lenkradw_Geschw = qnanf();
    uint8_t flex_LWI_01_LWI_VZ_Lenkradw_Geschw = 0;

    float flex_LH_EPS_03_EPS_Lenkmoment = qnanf();
    uint8_t flex_LH_EPS_03_EPS_VZ_Lenkmoment = 0;
    float flex_Klima_Sensor_02_BCM1_Aussen_Temp_ungef = qnanf();

    float flex_Motor_20_MO_Fahrpedalrohwert_01 = qnanf();
    uint8_t flex_Bremse_EV_01_EBKV_Fahrer_bremst = 0;
    float flex_Bremse_EV_01_EBKV_Bremspedalweg = qnanf();
    float flex_ESP_05_ESP_Bremsdruck = qnanf();
    uint8_t flex_Motor_14_MO_BLS = 0;
    uint8_t flex_Getriebe_11_GE_Zielgang = 0;

    float flex_ESP_03_ESP_VL_Radgeschw = qnanf(), flex_ESP_03_ESP_VR_Radgeschw = qnanf(),
        flex_ESP_03_ESP_HL_Radgeschw = qnanf(), flex_ESP_03_ESP_HR_Radgeschw = qnanf();
};

struct AdmaSnapshot {
    timespec ts{};

    double adma_ins_roll = qnand();
    double adma_ins_pitch = qnand();
    double adma_ins_yaw = qnand();

    double adma_ins_vel_hor_x = qnand(), adma_ins_vel_hor_y = qnand(), adma_ins_vel_hor_z = qnand();
    double adma_ins_vel_frame_x = qnand(), adma_ins_vel_frame_y = qnand(), adma_ins_vel_frame_z = qnand();
    double adma_ins_vel_hor_poi1_x = qnand(), adma_ins_vel_hor_poi1_y = qnand(), adma_ins_vel_hor_poi1_z = qnand();
    double adma_ins_vel_hor_poi2_x = qnand(), adma_ins_vel_hor_poi2_y = qnand(), adma_ins_vel_hor_poi2_z = qnand();
    double adma_gnss_vel_frame_x = qnand(), adma_gnss_vel_frame_y = qnand(), adma_gnss_vel_frame_z = qnand();

    double adma_acc_body_x = qnand(), adma_acc_body_y = qnand(), adma_acc_body_z = qnand();
    double adma_acc_horizontal_x = qnand(), adma_acc_horizontal_y = qnand(), adma_acc_horizontal_z = qnand();
    double adma_acc_body_poi1_x = qnand(), adma_acc_body_poi1_y = qnand(), adma_acc_body_poi1_z = qnand();
    double adma_acc_body_poi2_x = qnand(), adma_acc_body_poi2_y = qnand(), adma_acc_body_poi2_z = qnand();
    double adma_acc_horizontal_poi1_x = qnand(), adma_acc_horizontal_poi1_y = qnand(), adma_acc_horizontal_poi1_z = qnand();
    double adma_acc_horizontal_poi2_x = qnand(), adma_acc_horizontal_poi2_y = qnand(), adma_acc_horizontal_poi2_z = qnand();

    double adma_rates_body_x = qnand(), adma_rates_body_y = qnand(), adma_rates_body_z = qnand();
    double adma_rates_horizontal_x = qnand(), adma_rates_horizontal_y = qnand(), adma_rates_horizontal_z = qnand();

    double adma_misc_side_slip_angle = qnand();
    double adma_misc_distance_traveled = qnand();
    double adma_misc_poi1_side_slip_angle = qnand();
    double adma_misc_poi1_distance_traveled = qnand();
    double adma_misc_poi2_side_slip_angle = qnand();
    double adma_misc_poi2_distance_traveled = qnand();

    double adma_ins_pos_lat = qnand(), adma_ins_pos_lon = qnand(), adma_ins_height = qnand();
    double adma_ins_pos_poi1_lat = qnand(), adma_ins_pos_poi1_lon = qnand(), adma_ins_height_poi1 = qnand();
    double adma_ins_pos_poi2_lat = qnand(), adma_ins_pos_poi2_lon = qnand(), adma_ins_height_poi2 = qnand();

    int adma_gnss_sats_used = -1;
    int adma_gnss_sats_visible = -1;
    int adma_kf_status = -1;
    int adma_kf_lat_stimulated = -1;
    int adma_kf_long_stimulated = -1;
    int adma_kf_steady_state = -1;
};

struct CameraSnapshot {
    timespec ts{};
    std::string image_path;
    uint64_t frame_idx = 0;
};

struct SharedState {
    std::mutex mtx;
    std::condition_variable cv;
    FlexSnapshot flex;
    AdmaSnapshot adma;
    CameraSnapshot camera;
    uint64_t flex_seq = 0;
};

static bool decode_flex_packet(const unsigned char* buf, int n, const FlexSnapshot& prev, FlexSnapshot& out)
{
    if (n < 3) return false;

    out = prev;
    out.ts = now_realtime();
    out.cluster = buf[0];

    if (out.cluster < 1 || out.cluster > 64) return false;

    const unsigned char* pdus = buf + 3;
    const ClusterSignalOffsets& off = kClusterSignalOffsets[out.cluster];

    if (off.sara06 >= 0) {
        SARA_06 s06;
        s06.decode(pdus + off.sara06);
        out.flex_SARA_06_SARA_Accel_X_010 = s06.data().accel_x;
        out.flex_SARA_06_SARA_Accel_Y_010 = s06.data().accel_y;
        out.flex_SARA_06_SARA_Omega_Z_010 = s06.data().omega_z;
    }

    if (off.sara10 >= 0) {
        SARA_10 s10;
        s10.decode(pdus + off.sara10);
        out.flex_SARA_10_SARA_Accel_X_b = s10.data().accel_x;
        out.flex_SARA_10_SARA_Accel_Y_b = s10.data().accel_y;
        out.flex_SARA_10_SARA_Omega_Z_b = s10.data().omega_z;
    }

    if (off.esp21 >= 0) {
        ESP21 e;
        e.decode(pdus + off.esp21);
        out.flex_ESP_21_ESP_v_Signal = e.data().vehicle_speed;
        out.flex_ESP_21_ESP_Eingriff = e.data().esp_intervention;
    }

    if (off.esp03 >= 0) {
        ESP03 e3;
        e3.decode(pdus + off.esp03);
        out.flex_ESP_03_ESP_VL_Radgeschw = e3.data().wheel_speed_fl;
        out.flex_ESP_03_ESP_VR_Radgeschw = e3.data().wheel_speed_fr;
        out.flex_ESP_03_ESP_HL_Radgeschw = e3.data().wheel_speed_rl;
        out.flex_ESP_03_ESP_HR_Radgeschw = e3.data().wheel_speed_rr;
    }

    if (off.esp05 >= 0) {
        ESP05 e5;
        e5.decode(pdus + off.esp05);
        out.flex_ESP_05_ESP_Bremsdruck = e5.data().brake_pressure;
    }

    if (off.lwi01 >= 0) {
        LWI01 lwi;
        lwi.decode(pdus + off.lwi01);
        out.flex_LWI_01_LWI_Lenkradwinkel = lwi.angle;
        out.flex_LWI_01_LWI_VZ_Lenkradwinkel = lwi.angle_sign;
        out.flex_LWI_01_LWI_Lenkradw_Geschw = lwi.speed;
        out.flex_LWI_01_LWI_VZ_Lenkradw_Geschw = lwi.speed_sign;
    }

    if (off.lheps03 >= 0) {
        LHEPS03 eps;
        eps.decode(pdus + off.lheps03);
        out.flex_LH_EPS_03_EPS_Lenkmoment = eps.data().steering_torque;
        out.flex_LH_EPS_03_EPS_VZ_Lenkmoment = eps.data().steering_torque_sign;
    }

    if (off.klima_sensor_02 >= 0) {
        KlimaSensor02 k;
        k.decode(pdus + off.klima_sensor_02);
        out.flex_Klima_Sensor_02_BCM1_Aussen_Temp_ungef = k.data().external_temperature;
    }

    if (off.motor20 >= 0) {
        Motor20 m20;
        m20.decode(pdus + off.motor20);
        out.flex_Motor_20_MO_Fahrpedalrohwert_01 = m20.data().gas_percent;
    }

    if (off.bremse_ev01 >= 0) {
        BrakeEV01 br;
        br.decode(pdus + off.bremse_ev01);
        out.flex_Bremse_EV_01_EBKV_Fahrer_bremst = br.driver_brakes;
        out.flex_Bremse_EV_01_EBKV_Bremspedalweg = br.pedal_position;
    }

    if (off.motor14 >= 0) {
        Motor14 m14;
        m14.decode(pdus + off.motor14);
        out.flex_Motor_14_MO_BLS = m14.data().mo_bls;
    }

    const int getriebe11_offset = kGetriebe11Offsets[out.cluster];
    if (getriebe11_offset >= 0) {
        Getriebe11 getriebe11;
        getriebe11.decode(pdus + getriebe11_offset);
        out.flex_Getriebe_11_GE_Zielgang = getriebe11.data().ge_zielgang;
    }

    return true;
}

int main(int argc, char** argv)
{
    // -------- Output folders --------
    const std::filesystem::path run_log_dir = create_run_log_dir();
    const std::filesystem::path images_dir = run_log_dir / "images";
    const std::filesystem::path telemetry_csv_path = run_log_dir / "telemetry.csv";
    std::cerr << "Run log directory: " << run_log_dir.string() << "\n";

    // -------- CSV log --------
    std::ofstream csv(telemetry_csv_path.string());
    if (!csv) {
        std::cerr << "ERROR: cannot open " << telemetry_csv_path.string() << "\n";
        return 1;
    }
    csv << "idx,unix_sec,unix_nsec,cluster,"
            "flex_ESP_21_ESP_v_Signal,flex_SARA_06_SARA_Accel_X_010,flex_SARA_10_SARA_Accel_X_b,"
            "flex_SARA_06_SARA_Accel_Y_010,flex_SARA_10_SARA_Accel_Y_b,flex_SARA_06_SARA_Omega_Z_010,flex_SARA_10_SARA_Omega_Z_b,"
            "flex_LWI_01_LWI_Lenkradwinkel,flex_LWI_01_LWI_VZ_Lenkradwinkel,"
            "flex_LWI_01_LWI_Lenkradw_Geschw,flex_LWI_01_LWI_VZ_Lenkradw_Geschw,"
            "flex_LH_EPS_03_EPS_Lenkmoment,flex_LH_EPS_03_EPS_VZ_Lenkmoment,flex_Klima_Sensor_02_BCM1_Aussen_Temp_ungef,"
            "flex_Motor_20_MO_Fahrpedalrohwert_01,flex_Bremse_EV_01_EBKV_Fahrer_bremst,flex_Bremse_EV_01_EBKV_Bremspedalweg,"
            "flex_ESP_05_ESP_Bremsdruck,flex_Motor_14_MO_BLS,flex_Getriebe_11_GE_Zielgang,"
            "flex_ESP_03_ESP_VL_Radgeschw,flex_ESP_03_ESP_VR_Radgeschw,flex_ESP_03_ESP_HL_Radgeschw,flex_ESP_03_ESP_HR_Radgeschw,"
            "flex_ESP_21_ESP_Eingriff,"
            "adma_ins_vel_hor_x,adma_ins_vel_hor_y,adma_ins_vel_hor_z,"
            "adma_ins_vel_frame_x,adma_ins_vel_frame_y,adma_ins_vel_frame_z,"
            "adma_ins_vel_hor_poi1_x,adma_ins_vel_hor_poi1_y,adma_ins_vel_hor_poi1_z,"
            "adma_ins_vel_hor_poi2_x,adma_ins_vel_hor_poi2_y,adma_ins_vel_hor_poi2_z,"
            "adma_gnss_vel_frame_x,adma_gnss_vel_frame_y,adma_gnss_vel_frame_z,"
            "adma_acc_body_x,adma_acc_body_y,adma_acc_body_z,"
            "adma_acc_horizontal_x,adma_acc_horizontal_y,adma_acc_horizontal_z,"
            "adma_acc_body_poi1_x,adma_acc_body_poi1_y,adma_acc_body_poi1_z,"
            "adma_acc_body_poi2_x,adma_acc_body_poi2_y,adma_acc_body_poi2_z,"
            "adma_acc_horizontal_poi1_x,adma_acc_horizontal_poi1_y,adma_acc_horizontal_poi1_z,"
            "adma_acc_horizontal_poi2_x,adma_acc_horizontal_poi2_y,adma_acc_horizontal_poi2_z,"
            "adma_ins_roll,adma_ins_pitch,adma_ins_yaw,"
            "adma_rates_body_x,adma_rates_body_y,adma_rates_body_z,"
            "adma_rates_horizontal_x,adma_rates_horizontal_y,adma_rates_horizontal_z,"
            "adma_misc_side_slip_angle,adma_misc_distance_traveled,"
            "adma_misc_poi1_side_slip_angle,adma_misc_poi1_distance_traveled,"
            "adma_misc_poi2_side_slip_angle,adma_misc_poi2_distance_traveled,"
            "adma_ins_pos_lat,adma_ins_pos_lon,adma_ins_height,"
            "adma_ins_pos_poi1_lat,adma_ins_pos_poi1_lon,adma_ins_height_poi1,"
            "adma_ins_pos_poi2_lat,adma_ins_pos_poi2_lon,adma_ins_height_poi2,"
            "adma_gnss_sats_used,adma_gnss_sats_visible,"
            "adma_kf_status,adma_kf_lat_stimulated,adma_kf_long_stimulated,adma_kf_steady_state,"
            "image\n";
    csv.flush();

    // -------- Parse args --------
    Opts opts{};
    if (!parse_args(argc, argv, opts)) return 1;

    // -------- GStreamer init --------
    gst_init(&argc, &argv);

    SharedState shared{};

    std::thread flex_thread([&shared]() {
        UdpReceiver receiver(1500);
        unsigned char buf[65536];
        FlexSnapshot last_flex{};
        last_flex.ts = now_realtime();
        last_flex.cluster = 0;

        while (true) {
            const int n = receiver.receive(buf, sizeof(buf));
            if (n < 3) continue;

            FlexSnapshot snap;
            if (!decode_flex_packet(buf, n, last_flex, snap)) continue;
            last_flex = snap;

            {
                std::lock_guard<std::mutex> lk(shared.mtx);
                shared.flex = snap;
                ++shared.flex_seq;
            }
            shared.cv.notify_one();
        }
    });

    std::thread adma_thread([&shared]() {
        adma::AdmaPacketDecoder adma_decoder(adma::ProtocolVersion::V334);
        adma::AdmaUdpReceiver adma_receiver(
            "192.168.1.20",
            static_cast<uint16_t>(1021),
            std::optional<std::string>{"192.168.1.55"});

        while (true) {
            try {
                const auto adma_payload = adma_receiver.receive();
                const auto decoded = adma_decoder.decode(adma_payload);
                if (!decoded.v334.has_value()) continue;

                const auto& adma_packet = decoded.v334.value();
                AdmaSnapshot snap;
                snap.ts = now_realtime();

                snap.adma_ins_roll = static_cast<double>(adma_packet.insroll) * 0.01;
                snap.adma_ins_pitch = static_cast<double>(adma_packet.inspitch) * 0.01;
                snap.adma_ins_yaw = static_cast<double>(adma_packet.insyaw) * 0.01;

                snap.adma_ins_vel_hor_x = static_cast<double>(adma_packet.insVelHor.x) * 0.005;
                snap.adma_ins_vel_hor_y = static_cast<double>(adma_packet.insVelHor.y) * 0.005;
                snap.adma_ins_vel_hor_z = static_cast<double>(adma_packet.insVelHor.z) * 0.005;

                snap.adma_ins_vel_frame_x = static_cast<double>(adma_packet.insVelFrame.x) * 0.005;
                snap.adma_ins_vel_frame_y = static_cast<double>(adma_packet.insVelFrame.y) * 0.005;
                snap.adma_ins_vel_frame_z = static_cast<double>(adma_packet.insVelFrame.z) * 0.005;

                snap.adma_ins_vel_hor_poi1_x = static_cast<double>(adma_packet.insVelHorPOI[0].x) * 0.005;
                snap.adma_ins_vel_hor_poi1_y = static_cast<double>(adma_packet.insVelHorPOI[0].y) * 0.005;
                snap.adma_ins_vel_hor_poi1_z = static_cast<double>(adma_packet.insVelHorPOI[0].z) * 0.005;
                snap.adma_ins_vel_hor_poi2_x = static_cast<double>(adma_packet.insVelHorPOI[1].x) * 0.005;
                snap.adma_ins_vel_hor_poi2_y = static_cast<double>(adma_packet.insVelHorPOI[1].y) * 0.005;
                snap.adma_ins_vel_hor_poi2_z = static_cast<double>(adma_packet.insVelHorPOI[1].z) * 0.005;

                snap.adma_gnss_vel_frame_x = static_cast<double>(adma_packet.gnssvelframex) * 0.005;
                snap.adma_gnss_vel_frame_y = static_cast<double>(adma_packet.gnssvelframey) * 0.005;
                snap.adma_gnss_vel_frame_z = static_cast<double>(adma_packet.gnssvelframez) * 0.005;

                snap.adma_acc_body_x = static_cast<double>(adma_packet.accBody.x) * 0.0004;
                snap.adma_acc_body_y = static_cast<double>(adma_packet.accBody.y) * 0.0004;
                snap.adma_acc_body_z = static_cast<double>(adma_packet.accBody.z) * 0.0004;

                snap.adma_acc_horizontal_x = static_cast<double>(adma_packet.accHorizontal.x) * 0.0004;
                snap.adma_acc_horizontal_y = static_cast<double>(adma_packet.accHorizontal.y) * 0.0004;
                snap.adma_acc_horizontal_z = static_cast<double>(adma_packet.accHorizontal.z) * 0.0004;

                snap.adma_acc_body_poi1_x = static_cast<double>(adma_packet.accBodyPOI[0].x) * 0.0004;
                snap.adma_acc_body_poi1_y = static_cast<double>(adma_packet.accBodyPOI[0].y) * 0.0004;
                snap.adma_acc_body_poi1_z = static_cast<double>(adma_packet.accBodyPOI[0].z) * 0.0004;
                snap.adma_acc_body_poi2_x = static_cast<double>(adma_packet.accBodyPOI[1].x) * 0.0004;
                snap.adma_acc_body_poi2_y = static_cast<double>(adma_packet.accBodyPOI[1].y) * 0.0004;
                snap.adma_acc_body_poi2_z = static_cast<double>(adma_packet.accBodyPOI[1].z) * 0.0004;

                snap.adma_acc_horizontal_poi1_x = static_cast<double>(adma_packet.accHorizontalPOI[0].x) * 0.0004;
                snap.adma_acc_horizontal_poi1_y = static_cast<double>(adma_packet.accHorizontalPOI[0].y) * 0.0004;
                snap.adma_acc_horizontal_poi1_z = static_cast<double>(adma_packet.accHorizontalPOI[0].z) * 0.0004;
                snap.adma_acc_horizontal_poi2_x = static_cast<double>(adma_packet.accHorizontalPOI[1].x) * 0.0004;
                snap.adma_acc_horizontal_poi2_y = static_cast<double>(adma_packet.accHorizontalPOI[1].y) * 0.0004;
                snap.adma_acc_horizontal_poi2_z = static_cast<double>(adma_packet.accHorizontalPOI[1].z) * 0.0004;

                snap.adma_rates_body_x = static_cast<double>(adma_packet.ratesBody.x);
                snap.adma_rates_body_y = static_cast<double>(adma_packet.ratesBody.y);
                snap.adma_rates_body_z = static_cast<double>(adma_packet.ratesBody.z);

                snap.adma_rates_horizontal_x = static_cast<double>(adma_packet.ratesHorizontal.x);
                snap.adma_rates_horizontal_y = static_cast<double>(adma_packet.ratesHorizontal.y);
                snap.adma_rates_horizontal_z = static_cast<double>(adma_packet.ratesHorizontal.z);

                snap.adma_misc_side_slip_angle = static_cast<double>(adma_packet.misc.sideSlipAngle);
                snap.adma_misc_distance_traveled = static_cast<double>(adma_packet.misc.distanceTraveled);
                snap.adma_misc_poi1_side_slip_angle = static_cast<double>(adma_packet.miscPOI[0].sideSlipAngle);
                snap.adma_misc_poi1_distance_traveled = static_cast<double>(adma_packet.miscPOI[0].distanceTraveled);
                snap.adma_misc_poi2_side_slip_angle = static_cast<double>(adma_packet.miscPOI[1].sideSlipAngle);
                snap.adma_misc_poi2_distance_traveled = static_cast<double>(adma_packet.miscPOI[1].distanceTraveled);

                snap.adma_ins_pos_lat = static_cast<double>(adma_packet.insPos.pos_abs.latitude);
                snap.adma_ins_pos_lon = static_cast<double>(adma_packet.insPos.pos_abs.longitude);
                snap.adma_ins_height = static_cast<double>(adma_packet.insHeight);

                snap.adma_ins_pos_poi1_lat = static_cast<double>(adma_packet.insPosPOI[0].pos_abs.latitude);
                snap.adma_ins_pos_poi1_lon = static_cast<double>(adma_packet.insPosPOI[0].pos_abs.longitude);
                snap.adma_ins_height_poi1 = static_cast<double>(adma_packet.insHeightPOI[0]);
                snap.adma_ins_pos_poi2_lat = static_cast<double>(adma_packet.insPosPOI[1].pos_abs.latitude);
                snap.adma_ins_pos_poi2_lon = static_cast<double>(adma_packet.insPosPOI[1].pos_abs.longitude);
                snap.adma_ins_height_poi2 = static_cast<double>(adma_packet.insHeightPOI[1]);

                snap.adma_gnss_sats_used = static_cast<int>(adma_packet.gnsssatsused);
                snap.adma_gnss_sats_visible = static_cast<int>(adma_packet.gnsssatsvisible);
                snap.adma_kf_status = static_cast<int>(adma_packet.kfStatus);
                snap.adma_kf_lat_stimulated = static_cast<int>(adma_packet.kflatstimulated);
                snap.adma_kf_long_stimulated = static_cast<int>(adma_packet.kflongstimulated);
                snap.adma_kf_steady_state = static_cast<int>(adma_packet.kfsteadystate);

                {
                    std::lock_guard<std::mutex> lk(shared.mtx);
                    shared.adma = snap;
                }
            } catch (const std::exception& ex) {
                std::cerr << "WARN: ADMA receive/decode failed: " << ex.what() << "\n";
            }
        }
    });

    std::thread camera_thread([&shared, &opts, &images_dir]() {
        const std::string pipeline_desc = build_pipeline(opts);
        std::cerr << "pipeline:\n  " << pipeline_desc << "\n";

        GError* err = nullptr;
        GstElement* pipeline = gst_parse_launch(pipeline_desc.c_str(), &err);
        if (!pipeline) {
            std::cerr << "ERROR: gst_parse_launch failed\n";
            if (err) { std::cerr << "  " << err->message << "\n"; g_error_free(err); }
            return;
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
            return;
        }
        GstAppSink* appsink = GST_APP_SINK(sink_elem);

        if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "ERROR: failed to set GStreamer pipeline to PLAYING\n";
            gst_object_unref(sink_elem);
            gst_object_unref(pipeline);
            return;
        }

        uint64_t cam_idx = 0;
        while (true) {
            if (!check_bus_nonblocking(pipeline)) {
                std::cerr << "Stopping camera thread due to GStreamer error/EOS.\n";
                break;
            }

            GstSample* sample = gst_app_sink_try_pull_sample(appsink, 100000000ULL);
            if (!sample) continue;

            ++cam_idx;
            char img_name[32];
            std::snprintf(img_name, sizeof(img_name), "%06llu.jpg",
                          static_cast<unsigned long long>(cam_idx));
            const std::filesystem::path img_path = images_dir / img_name;

            if (!save_jpeg_from_sample(sample, img_path.string().c_str())) {
                std::cerr << "WARN: failed to save camera frame idx=" << cam_idx << "\n";
                gst_sample_unref(sample);
                continue;
            }

            CameraSnapshot cam;
            cam.ts = now_realtime();
            cam.image_path = img_path.string();
            cam.frame_idx = cam_idx;

            {
                std::lock_guard<std::mutex> lk(shared.mtx);
                shared.camera = std::move(cam);
            }

            gst_sample_unref(sample);
        }

        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(sink_elem);
        gst_object_unref(pipeline);
    });

    uint64_t idx = 0;
    uint64_t consumed_flex_seq = 0;

    while (true) {
        FlexSnapshot flex;
        AdmaSnapshot adma;
        CameraSnapshot cam;

        {
            std::unique_lock<std::mutex> lk(shared.mtx);
            shared.cv.wait(lk, [&]() { return shared.flex_seq != consumed_flex_seq; });
            consumed_flex_seq = shared.flex_seq;
            flex = shared.flex;
            adma = shared.adma;
            cam = shared.camera;
        }

        ++idx;

        print_line(adma.adma_kf_status,
               adma.adma_kf_lat_stimulated,
               adma.adma_kf_long_stimulated,
               adma.adma_kf_steady_state,
               adma.adma_ins_vel_hor_x,
               adma.adma_ins_vel_hor_y,
               adma.adma_acc_body_y,
             flex.flex_LWI_01_LWI_Lenkradwinkel,
                   flex.ts);

        csv << idx << ","
            << flex.ts.tv_sec << "," << flex.ts.tv_nsec << ","
            << flex.cluster << ","
            << flex.flex_ESP_21_ESP_v_Signal << ","
            << flex.flex_SARA_06_SARA_Accel_X_010 << "," << flex.flex_SARA_10_SARA_Accel_X_b << ","
            << flex.flex_SARA_06_SARA_Accel_Y_010 << "," << flex.flex_SARA_10_SARA_Accel_Y_b << ","
            << flex.flex_SARA_06_SARA_Omega_Z_010 << "," << flex.flex_SARA_10_SARA_Omega_Z_b << ","
            << flex.flex_LWI_01_LWI_Lenkradwinkel << "," << static_cast<unsigned>(flex.flex_LWI_01_LWI_VZ_Lenkradwinkel) << ","
            << flex.flex_LWI_01_LWI_Lenkradw_Geschw << "," << static_cast<unsigned>(flex.flex_LWI_01_LWI_VZ_Lenkradw_Geschw) << ","
            << flex.flex_LH_EPS_03_EPS_Lenkmoment << "," << static_cast<unsigned>(flex.flex_LH_EPS_03_EPS_VZ_Lenkmoment) << ","
            << flex.flex_Klima_Sensor_02_BCM1_Aussen_Temp_ungef << ","
            << flex.flex_Motor_20_MO_Fahrpedalrohwert_01 << "," << static_cast<unsigned>(flex.flex_Bremse_EV_01_EBKV_Fahrer_bremst) << ","
            << flex.flex_Bremse_EV_01_EBKV_Bremspedalweg << "," << flex.flex_ESP_05_ESP_Bremsdruck << ","
            << static_cast<unsigned>(flex.flex_Motor_14_MO_BLS) << ","
            << static_cast<unsigned>(flex.flex_Getriebe_11_GE_Zielgang) << ","
            << flex.flex_ESP_03_ESP_VL_Radgeschw << "," << flex.flex_ESP_03_ESP_VR_Radgeschw << ","
            << flex.flex_ESP_03_ESP_HL_Radgeschw << "," << flex.flex_ESP_03_ESP_HR_Radgeschw << ","
            << static_cast<unsigned>(flex.flex_ESP_21_ESP_Eingriff) << ","
            << adma.adma_ins_vel_hor_x << "," << adma.adma_ins_vel_hor_y << "," << adma.adma_ins_vel_hor_z << ","
            << adma.adma_ins_vel_frame_x << "," << adma.adma_ins_vel_frame_y << "," << adma.adma_ins_vel_frame_z << ","
            << adma.adma_ins_vel_hor_poi1_x << "," << adma.adma_ins_vel_hor_poi1_y << "," << adma.adma_ins_vel_hor_poi1_z << ","
            << adma.adma_ins_vel_hor_poi2_x << "," << adma.adma_ins_vel_hor_poi2_y << "," << adma.adma_ins_vel_hor_poi2_z << ","
            << adma.adma_gnss_vel_frame_x << "," << adma.adma_gnss_vel_frame_y << "," << adma.adma_gnss_vel_frame_z << ","
            << adma.adma_acc_body_x << "," << adma.adma_acc_body_y << "," << adma.adma_acc_body_z << ","
            << adma.adma_acc_horizontal_x << "," << adma.adma_acc_horizontal_y << "," << adma.adma_acc_horizontal_z << ","
            << adma.adma_acc_body_poi1_x << "," << adma.adma_acc_body_poi1_y << "," << adma.adma_acc_body_poi1_z << ","
            << adma.adma_acc_body_poi2_x << "," << adma.adma_acc_body_poi2_y << "," << adma.adma_acc_body_poi2_z << ","
            << adma.adma_acc_horizontal_poi1_x << "," << adma.adma_acc_horizontal_poi1_y << "," << adma.adma_acc_horizontal_poi1_z << ","
            << adma.adma_acc_horizontal_poi2_x << "," << adma.adma_acc_horizontal_poi2_y << "," << adma.adma_acc_horizontal_poi2_z << ","
            << adma.adma_ins_roll << "," << adma.adma_ins_pitch << "," << adma.adma_ins_yaw << ","
            << adma.adma_rates_body_x << "," << adma.adma_rates_body_y << "," << adma.adma_rates_body_z << ","
            << adma.adma_rates_horizontal_x << "," << adma.adma_rates_horizontal_y << "," << adma.adma_rates_horizontal_z << ","
            << adma.adma_misc_side_slip_angle << "," << adma.adma_misc_distance_traveled << ","
            << adma.adma_misc_poi1_side_slip_angle << "," << adma.adma_misc_poi1_distance_traveled << ","
            << adma.adma_misc_poi2_side_slip_angle << "," << adma.adma_misc_poi2_distance_traveled << ","
            << adma.adma_ins_pos_lat << "," << adma.adma_ins_pos_lon << "," << adma.adma_ins_height << ","
            << adma.adma_ins_pos_poi1_lat << "," << adma.adma_ins_pos_poi1_lon << "," << adma.adma_ins_height_poi1 << ","
            << adma.adma_ins_pos_poi2_lat << "," << adma.adma_ins_pos_poi2_lon << "," << adma.adma_ins_height_poi2 << ","
            << adma.adma_gnss_sats_used << "," << adma.adma_gnss_sats_visible << ","
            << adma.adma_kf_status << "," << adma.adma_kf_lat_stimulated << "," << adma.adma_kf_long_stimulated << "," << adma.adma_kf_steady_state << ","
            << cam.image_path << "\n";
        csv.flush();
    }

    flex_thread.join();
    adma_thread.join();
    camera_thread.join();
    return 0;
}
