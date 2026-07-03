// udp_decoder_gst.cpp
// UDP/JSON (Automotive Ethernet) signals + dual camera capture via GStreamer ->
// save idx.jpg -> log CSV row with same idx.
//
// The incoming UDP JSON contains ETH signals listed in
// AI-chassis-control-pipeline/windows_udp_server/eth_signals.txt, pushed by
// AI-chassis-control-pipeline/windows_udp_server/main.py running on a Windows
// PC, which sends a flat JSON UDP datagram with prefixed keys, e.g.:
//   {"eth_LWI_AgStgWhl": -0.5, ...}
//
// Build (example):
//   g++ -O2 -std=c++17 main.cpp -o udp_decoder `pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0`
//
// Run examples:
//   ./udp_decoder --device /dev/video0 --w 2560 --h 720 --fps 30 --crop right --crop-px 1280
//   ./udp_decoder --device /dev/video0 --w 1280 --h 720 --fps 60 --crop right --crop-px 640
//   ./udp_decoder --device /dev/video0 --w 672  --h 376 --fps 100 --crop none

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <array>
#include <vector>
#include <optional>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <csignal>

#include "UdpReceiver.hpp"
#include "AdmaDecoder.hpp"
#include "AdmaUdpReceiver.hpp"

struct VectorSnapshot {
    timespec ts{};

    double VDSO_Vx3dKmph = 0.0;
    double IMU_ALat = 0.0;
    double IMU_ALgt = 0.0;
    double IMU_YawRate = 0.0;
    double LWI_AgStgWhl = 0.0;
    double EPS_RackPosnSpd = 0.0;
    double EPS_SpdResv = 0.0;
    double EPS_StgTq = 0.0;
    double OTP_Aussen_Temp_ungef = 0.0;
    double API_PrctDrvPed = 0.0;
    double API_StDrvPed = 0.0;
    double DBR_PrctLPosnBrkPedDrv = 0.0;
    double DBR_LPosnBrkPedDrv = 0.0;
    double DBR_FBrkPedDrv = 0.0;
    double VMM_BrkPed_ItDmdNorm = 0.0;
    double BLC_BrkLghtReq_XIX_ESC_03_XIX_VLAN_FAS = 0.0;
    double DBR_IdcDrvBrk_XIX_ESC_06_XIX_VLAN_FAS = 0.0;
    double DPM_StDispDrvPosn_XIX_HCP1_15_XIX_VLAN_FAS = 0.0;
    double VDSO_VWhlSpdFrLe = 0.0;
    double VDSO_VWhlSpdFrRi = 0.0;
    double VDSO_VWhlSpdReLe = 0.0;
    double VDSO_VWhlSpdReRi = 0.0;
    double VDC_Intv = 0.0;
    double VDSO_AgVehSideSlip = 0.0;

    double POS_GNSS_Breite_Ortung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS = 0.0;
    double POS_GNSS_Laenge_Ortung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS = 0.0;
    double POS_GNSS_Ortung_Ausrichtung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS = 0.0;
    double POS_GNSS_Sichtbare_Satelliten_XIX_POS_GNSS_07_Sat_konf_XIX_VLAN_FAS = 0.0;
    double POS_GNSS_Ortung_Hoehe_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS = 0.0;
};

struct VectorSignalBinding {
    const char* json_key;
    const char* field_name;
    double VectorSnapshot::*member;
};

static const std::array<VectorSignalBinding, 29> kVectorSignalBindings = {{
    {"eth_VDSO_Vx3dKmph", "VDSO_Vx3dKmph", &VectorSnapshot::VDSO_Vx3dKmph},
    {"eth_IMU_ALat", "IMU_ALat", &VectorSnapshot::IMU_ALat},
    {"eth_IMU_ALgt", "IMU_ALgt", &VectorSnapshot::IMU_ALgt},
    {"eth_IMU_YawRate", "IMU_YawRate", &VectorSnapshot::IMU_YawRate},
    {"eth_LWI_AgStgWhl", "LWI_AgStgWhl", &VectorSnapshot::LWI_AgStgWhl},
    {"eth_EPS_RackPosnSpd", "EPS_RackPosnSpd", &VectorSnapshot::EPS_RackPosnSpd},
    {"eth_EPS_SpdResv", "EPS_SpdResv", &VectorSnapshot::EPS_SpdResv},
    {"eth_EPS_StgTq", "EPS_StgTq", &VectorSnapshot::EPS_StgTq},
    {"eth_OTP_Aussen_Temp_ungef", "OTP_Aussen_Temp_ungef", &VectorSnapshot::OTP_Aussen_Temp_ungef},
    {"eth_API_PrctDrvPed", "API_PrctDrvPed", &VectorSnapshot::API_PrctDrvPed},
    {"eth_API_StDrvPed", "API_StDrvPed", &VectorSnapshot::API_StDrvPed},
    {"eth_DBR_PrctLPosnBrkPedDrv", "DBR_PrctLPosnBrkPedDrv", &VectorSnapshot::DBR_PrctLPosnBrkPedDrv},
    {"eth_DBR_LPosnBrkPedDrv", "DBR_LPosnBrkPedDrv", &VectorSnapshot::DBR_LPosnBrkPedDrv},
    {"eth_DBR_FBrkPedDrv", "DBR_FBrkPedDrv", &VectorSnapshot::DBR_FBrkPedDrv},
    {"eth_VMM_BrkPed_ItDmdNorm", "VMM_BrkPed_ItDmdNorm", &VectorSnapshot::VMM_BrkPed_ItDmdNorm},
    {"eth_BLC_BrkLghtReq_XIX_ESC_03_XIX_VLAN_FAS", "BLC_BrkLghtReq_XIX_ESC_03_XIX_VLAN_FAS", &VectorSnapshot::BLC_BrkLghtReq_XIX_ESC_03_XIX_VLAN_FAS},
    {"eth_DBR_IdcDrvBrk_XIX_ESC_06_XIX_VLAN_FAS", "DBR_IdcDrvBrk_XIX_ESC_06_XIX_VLAN_FAS", &VectorSnapshot::DBR_IdcDrvBrk_XIX_ESC_06_XIX_VLAN_FAS},
    {"eth_DPM_StDispDrvPosn_XIX_HCP1_15_XIX_VLAN_FAS", "DPM_StDispDrvPosn_XIX_HCP1_15_XIX_VLAN_FAS", &VectorSnapshot::DPM_StDispDrvPosn_XIX_HCP1_15_XIX_VLAN_FAS},
    {"eth_VDSO_VWhlSpdFrLe", "VDSO_VWhlSpdFrLe", &VectorSnapshot::VDSO_VWhlSpdFrLe},
    {"eth_VDSO_VWhlSpdFrRi", "VDSO_VWhlSpdFrRi", &VectorSnapshot::VDSO_VWhlSpdFrRi},
    {"eth_VDSO_VWhlSpdReLe", "VDSO_VWhlSpdReLe", &VectorSnapshot::VDSO_VWhlSpdReLe},
    {"eth_VDSO_VWhlSpdReRi", "VDSO_VWhlSpdReRi", &VectorSnapshot::VDSO_VWhlSpdReRi},
    {"eth_VDC_Intv", "VDC_Intv", &VectorSnapshot::VDC_Intv},
    {"eth_VDSO_AgVehSideSlip", "VDSO_AgVehSideSlip", &VectorSnapshot::VDSO_AgVehSideSlip},
    {"eth_POS_GNSS_Breite_Ortung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS", "POS_GNSS_Breite_Ortung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS", &VectorSnapshot::POS_GNSS_Breite_Ortung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS},
    {"eth_POS_GNSS_Laenge_Ortung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS", "POS_GNSS_Laenge_Ortung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS", &VectorSnapshot::POS_GNSS_Laenge_Ortung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS},
    {"eth_POS_GNSS_Ortung_Ausrichtung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS", "POS_GNSS_Ortung_Ausrichtung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS", &VectorSnapshot::POS_GNSS_Ortung_Ausrichtung_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS},
    {"eth_POS_GNSS_Sichtbare_Satelliten_XIX_POS_GNSS_07_Sat_konf_XIX_VLAN_FAS", "POS_GNSS_Sichtbare_Satelliten_XIX_POS_GNSS_07_Sat_konf_XIX_VLAN_FAS", &VectorSnapshot::POS_GNSS_Sichtbare_Satelliten_XIX_POS_GNSS_07_Sat_konf_XIX_VLAN_FAS},
    {"eth_POS_GNSS_Ortung_Hoehe_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS", "POS_GNSS_Ortung_Hoehe_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS", &VectorSnapshot::POS_GNSS_Ortung_Hoehe_XIX_POS_GNSS_05_Position_XIX_VLAN_FAS},
}};

static bool set_vector_signal_value(VectorSnapshot& snapshot, const std::string& key, double value)
{
    for (const auto& binding : kVectorSignalBindings) {
        if (key == binding.json_key || key == binding.field_name) {
            snapshot.*(binding.member) = value;
            return true;
        }
    }
    return false;
}

static inline void print_line(
    int adma_kf_status,
    int adma_kf_lat_stimulated,
    int adma_kf_long_stimulated,
    int adma_kf_steady_state,
    double eth_LWI_AgStgWhl,
    double eth_VDSO_Vx3dKmph,
    const timespec& ts)
{
    std::tm tm_local{};
    localtime_r(&ts.tv_sec, &tm_local);

    char tb[32];
    std::strftime(tb, sizeof(tb), "%H:%M:%S", &tm_local);
    long ms = ts.tv_nsec / 1000000;

    std::cout << std::fixed << std::setprecision(3)
              << "timestamp: " << tb << "." << std::setw(3) << std::setfill('0') << ms
              << std::setfill(' ') << "\n"
              << "adma_kf_status = " << adma_kf_status << "\n"
              << "adma_kf_lat_stimulated = " << adma_kf_lat_stimulated << "\n"
              << "adma_kf_long_stimulated = " << adma_kf_long_stimulated << "\n"
              << "adma_kf_steady_state = " << adma_kf_steady_state << "\n"
              << "vector_snapshot.LWI_AgStgWhl = " << eth_LWI_AgStgWhl << "\n"
              << "vector_snapshot.VDSO_Vx3dKmph = " << eth_VDSO_Vx3dKmph << "\n"
              << "----------------------------------------\n";
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
    uint16_t eth_port = 5005; // UDP port for JSON datagrams from windows_udp_server/main.py
    std::string log_path;
};

static void usage(const char* argv0)
{
    std::cerr
        << "Usage:\n"
    << "  " << argv0 << " [--eth-port 5005] [--log_path ~/Desktop]\n\n"
        << "Examples:\n"
    << "  " << argv0 << " --eth-port 5005\n"
    << "  " << argv0 << " --log_path ~/Desktop\n"
    << "  " << argv0 << " --log_path=~/Desktop\n";
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

        if (a == "--eth-port" || a.rfind("--eth-port=", 0) == 0) {
            const char* v = nullptr;
            std::string v_inline;
            if (a.rfind("--eth-port=", 0) == 0) {
                v_inline = a.substr(std::strlen("--eth-port="));
                v = v_inline.c_str();
            } else {
                v = need("--eth-port");
            }
            if (!v) return false;
            o.eth_port = static_cast<uint16_t>(std::atoi(v));
        } else if (a == "--log_path" || a.rfind("--log_path=", 0) == 0) {
            const char* v = nullptr;
            std::string v_inline;
            if (a.rfind("--log_path=", 0) == 0) {
                v_inline = a.substr(std::strlen("--log_path="));
                v = v_inline.c_str();
            } else {
                v = need("--log_path");
            }
            if (!v) return false;
            o.log_path = v;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown arg: " << a << "\n";
            usage(argv[0]);
            return false;
        }
    }

    return true;
}

static std::string build_camera_pipeline(int sensor_id)
{
    std::ostringstream ss;
    ss << "nvarguscamerasrc sensor-id=" << sensor_id << " ! "
       << "video/x-raw(memory:NVMM),width=960,height=600,framerate=120/1 ! "
       << "queue ! "
       << "nvvidconv ! "
       << "nvjpegenc quality=85 ! "
       << "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true";

    return ss.str();
}

static inline timespec now_realtime()
{
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts;
}

static std::filesystem::path expand_tilde_path(const std::string& input_path)
{
    namespace fs = std::filesystem;
    if (input_path.empty() || input_path[0] != '~') {
        return fs::path(input_path);
    }

    const char* home = std::getenv("HOME");
    if (!home || !*home) home = std::getenv("USERPROFILE");

    std::string home_fallback;
    if ((!home || !*home)) {
        const char* drive = std::getenv("HOMEDRIVE");
        const char* path = std::getenv("HOMEPATH");
        if (drive && path) {
            home_fallback = std::string(drive) + path;
            home = home_fallback.c_str();
        }
    }

    if (!home || !*home) {
        return fs::path(input_path);
    }

    if (input_path.size() == 1) {
        return fs::path(home);
    }

    const char next = input_path[1];
    if (next != '/' && next != '\\') {
        // ~user style is not supported; keep the original path.
        return fs::path(input_path);
    }

    fs::path expanded(home);
    if (input_path.size() > 2) {
        expanded /= input_path.substr(2);
    }
    return expanded;
}

static std::filesystem::path create_run_log_dir(const std::filesystem::path& log_root_path)
{
    namespace fs = std::filesystem;

    const fs::path base_log_dir = log_root_path.empty()
        ? fs::path("log")
        : (log_root_path / "log");
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
            if (c < '0' || c > '9') { all_digits = false; break; }
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
    fs::create_directories(run_dir / "images" / "cam0");
    fs::create_directories(run_dir / "images" / "cam2");
    return run_dir;
}

struct AdmaSnapshot {
    timespec ts{};

    double adma_ins_roll = 0.0;
    double adma_ins_pitch = 0.0;
    double adma_ins_yaw = 0.0;

    double adma_ins_vel_hor_x = 0.0, adma_ins_vel_hor_y = 0.0, adma_ins_vel_hor_z = 0.0;
    double adma_ins_vel_frame_x = 0.0, adma_ins_vel_frame_y = 0.0, adma_ins_vel_frame_z = 0.0;
    double adma_ins_vel_hor_poi1_x = 0.0, adma_ins_vel_hor_poi1_y = 0.0, adma_ins_vel_hor_poi1_z = 0.0;
    double adma_ins_vel_hor_poi2_x = 0.0, adma_ins_vel_hor_poi2_y = 0.0, adma_ins_vel_hor_poi2_z = 0.0;
    double adma_ins_vel_hor_poi3_x = 0.0, adma_ins_vel_hor_poi3_y = 0.0, adma_ins_vel_hor_poi3_z = 0.0;
    double adma_gnss_vel_frame_x = 0.0, adma_gnss_vel_frame_y = 0.0, adma_gnss_vel_frame_z = 0.0;

    double adma_acc_body_x = 0.0, adma_acc_body_y = 0.0, adma_acc_body_z = 0.0;
    double adma_acc_horizontal_x = 0.0, adma_acc_horizontal_y = 0.0, adma_acc_horizontal_z = 0.0;
    double adma_acc_body_poi1_x = 0.0, adma_acc_body_poi1_y = 0.0, adma_acc_body_poi1_z = 0.0;
    double adma_acc_body_poi2_x = 0.0, adma_acc_body_poi2_y = 0.0, adma_acc_body_poi2_z = 0.0;
    double adma_acc_body_poi3_x = 0.0, adma_acc_body_poi3_y = 0.0, adma_acc_body_poi3_z = 0.0;
    double adma_acc_horizontal_poi1_x = 0.0, adma_acc_horizontal_poi1_y = 0.0, adma_acc_horizontal_poi1_z = 0.0;
    double adma_acc_horizontal_poi2_x = 0.0, adma_acc_horizontal_poi2_y = 0.0, adma_acc_horizontal_poi2_z = 0.0;
    double adma_acc_horizontal_poi3_x = 0.0, adma_acc_horizontal_poi3_y = 0.0, adma_acc_horizontal_poi3_z = 0.0;

    double adma_rates_body_x = 0.0, adma_rates_body_y = 0.0, adma_rates_body_z = 0.0;
    double adma_rates_horizontal_x = 0.0, adma_rates_horizontal_y = 0.0, adma_rates_horizontal_z = 0.0;

    double adma_misc_side_slip_angle = 0.0;
    double adma_misc_distance_traveled = 0.0;
    double adma_misc_poi1_side_slip_angle = 0.0;
    double adma_misc_poi1_distance_traveled = 0.0;
    double adma_misc_poi2_side_slip_angle = 0.0;
    double adma_misc_poi2_distance_traveled = 0.0;
    double adma_misc_poi3_side_slip_angle = 0.0;
    double adma_misc_poi3_distance_traveled = 0.0;

    double adma_ins_pos_lat = 0.0, adma_ins_pos_lon = 0.0, adma_ins_height = 0.0;
    double adma_ins_pos_poi1_lat = 0.0, adma_ins_pos_poi1_lon = 0.0, adma_ins_height_poi1 = 0.0;
    double adma_ins_pos_poi2_lat = 0.0, adma_ins_pos_poi2_lon = 0.0, adma_ins_height_poi2 = 0.0;
    double adma_ins_pos_poi3_lat = 0.0, adma_ins_pos_poi3_lon = 0.0, adma_ins_height_poi3 = 0.0;

    int adma_gnss_sats_used = -1;
    int adma_gnss_sats_visible = -1;
    int adma_kf_status = -1;
    int adma_kf_lat_stimulated = -1;
    int adma_kf_long_stimulated = -1;
    int adma_kf_steady_state = -1;
};

// CSV column names for the ADMA signals. Keep in sync with write_adma_csv_values().
static constexpr const char* kAdmaCsvHeader =
    "adma_ins_vel_hor_x,adma_ins_vel_hor_y,adma_ins_vel_hor_z,"
    "adma_ins_vel_frame_x,adma_ins_vel_frame_y,adma_ins_vel_frame_z,"
    "adma_ins_vel_hor_poi1_x,adma_ins_vel_hor_poi1_y,adma_ins_vel_hor_poi1_z,"
    "adma_ins_vel_hor_poi2_x,adma_ins_vel_hor_poi2_y,adma_ins_vel_hor_poi2_z,"
    "adma_ins_vel_hor_poi3_x,adma_ins_vel_hor_poi3_y,adma_ins_vel_hor_poi3_z,"
    "adma_gnss_vel_frame_x,adma_gnss_vel_frame_y,adma_gnss_vel_frame_z,"
    "adma_acc_body_x,adma_acc_body_y,adma_acc_body_z,"
    "adma_acc_horizontal_x,adma_acc_horizontal_y,adma_acc_horizontal_z,"
    "adma_acc_body_poi1_x,adma_acc_body_poi1_y,adma_acc_body_poi1_z,"
    "adma_acc_body_poi2_x,adma_acc_body_poi2_y,adma_acc_body_poi2_z,"
    "adma_acc_body_poi3_x,adma_acc_body_poi3_y,adma_acc_body_poi3_z,"
    "adma_acc_horizontal_poi1_x,adma_acc_horizontal_poi1_y,adma_acc_horizontal_poi1_z,"
    "adma_acc_horizontal_poi2_x,adma_acc_horizontal_poi2_y,adma_acc_horizontal_poi2_z,"
    "adma_acc_horizontal_poi3_x,adma_acc_horizontal_poi3_y,adma_acc_horizontal_poi3_z,"
    "adma_ins_roll,adma_ins_pitch,adma_ins_yaw,"
    "adma_rates_body_x,adma_rates_body_y,adma_rates_body_z,"
    "adma_rates_horizontal_x,adma_rates_horizontal_y,adma_rates_horizontal_z,"
    "adma_misc_side_slip_angle,adma_misc_distance_traveled,"
    "adma_misc_poi1_side_slip_angle,adma_misc_poi1_distance_traveled,"
    "adma_misc_poi2_side_slip_angle,adma_misc_poi2_distance_traveled,"
    "adma_misc_poi3_side_slip_angle,adma_misc_poi3_distance_traveled,"
    "adma_ins_pos_lat,adma_ins_pos_lon,adma_ins_height,"
    "adma_ins_pos_poi1_lat,adma_ins_pos_poi1_lon,adma_ins_height_poi1,"
    "adma_ins_pos_poi2_lat,adma_ins_pos_poi2_lon,adma_ins_height_poi2,"
    "adma_ins_pos_poi3_lat,adma_ins_pos_poi3_lon,adma_ins_height_poi3,"
    "adma_gnss_sats_used,adma_gnss_sats_visible,"
    "adma_kf_status,adma_kf_lat_stimulated,adma_kf_long_stimulated,adma_kf_steady_state";

// Writes the ADMA signal values in the same order as kAdmaCsvHeader.
static void write_adma_csv_values(std::ostream& csv, const AdmaSnapshot& adma)
{
    csv << adma.adma_ins_vel_hor_x << "," << adma.adma_ins_vel_hor_y << "," << adma.adma_ins_vel_hor_z << ","
        << adma.adma_ins_vel_frame_x << "," << adma.adma_ins_vel_frame_y << "," << adma.adma_ins_vel_frame_z << ","
        << adma.adma_ins_vel_hor_poi1_x << "," << adma.adma_ins_vel_hor_poi1_y << "," << adma.adma_ins_vel_hor_poi1_z << ","
        << adma.adma_ins_vel_hor_poi2_x << "," << adma.adma_ins_vel_hor_poi2_y << "," << adma.adma_ins_vel_hor_poi2_z << ","
        << adma.adma_ins_vel_hor_poi3_x << "," << adma.adma_ins_vel_hor_poi3_y << "," << adma.adma_ins_vel_hor_poi3_z << ","
        << adma.adma_gnss_vel_frame_x << "," << adma.adma_gnss_vel_frame_y << "," << adma.adma_gnss_vel_frame_z << ","
        << adma.adma_acc_body_x << "," << adma.adma_acc_body_y << "," << adma.adma_acc_body_z << ","
        << adma.adma_acc_horizontal_x << "," << adma.adma_acc_horizontal_y << "," << adma.adma_acc_horizontal_z << ","
        << adma.adma_acc_body_poi1_x << "," << adma.adma_acc_body_poi1_y << "," << adma.adma_acc_body_poi1_z << ","
        << adma.adma_acc_body_poi2_x << "," << adma.adma_acc_body_poi2_y << "," << adma.adma_acc_body_poi2_z << ","
        << adma.adma_acc_body_poi3_x << "," << adma.adma_acc_body_poi3_y << "," << adma.adma_acc_body_poi3_z << ","
        << adma.adma_acc_horizontal_poi1_x << "," << adma.adma_acc_horizontal_poi1_y << "," << adma.adma_acc_horizontal_poi1_z << ","
        << adma.adma_acc_horizontal_poi2_x << "," << adma.adma_acc_horizontal_poi2_y << "," << adma.adma_acc_horizontal_poi2_z << ","
        << adma.adma_acc_horizontal_poi3_x << "," << adma.adma_acc_horizontal_poi3_y << "," << adma.adma_acc_horizontal_poi3_z << ","
        << adma.adma_ins_roll << "," << adma.adma_ins_pitch << "," << adma.adma_ins_yaw << ","
        << adma.adma_rates_body_x << "," << adma.adma_rates_body_y << "," << adma.adma_rates_body_z << ","
        << adma.adma_rates_horizontal_x << "," << adma.adma_rates_horizontal_y << "," << adma.adma_rates_horizontal_z << ","
        << adma.adma_misc_side_slip_angle << "," << adma.adma_misc_distance_traveled << ","
        << adma.adma_misc_poi1_side_slip_angle << "," << adma.adma_misc_poi1_distance_traveled << ","
        << adma.adma_misc_poi2_side_slip_angle << "," << adma.adma_misc_poi2_distance_traveled << ","
        << adma.adma_misc_poi3_side_slip_angle << "," << adma.adma_misc_poi3_distance_traveled << ","
        << adma.adma_ins_pos_lat << "," << adma.adma_ins_pos_lon << "," << adma.adma_ins_height << ","
        << adma.adma_ins_pos_poi1_lat << "," << adma.adma_ins_pos_poi1_lon << "," << adma.adma_ins_height_poi1 << ","
        << adma.adma_ins_pos_poi2_lat << "," << adma.adma_ins_pos_poi2_lon << "," << adma.adma_ins_height_poi2 << ","
        << adma.adma_ins_pos_poi3_lat << "," << adma.adma_ins_pos_poi3_lon << "," << adma.adma_ins_height_poi3 << ","
        << adma.adma_gnss_sats_used << "," << adma.adma_gnss_sats_visible << ","
        << adma.adma_kf_status << "," << adma.adma_kf_lat_stimulated << ","
        << adma.adma_kf_long_stimulated << "," << adma.adma_kf_steady_state;
}

struct CameraSnapshot {
    timespec ts{};
    std::string image_path;
    uint64_t frame_idx = 0;
};

struct SharedState {
    std::mutex mtx;
    std::condition_variable cv;
    VectorSnapshot vector_snapshot;
    AdmaSnapshot adma;
    CameraSnapshot cam0;
    CameraSnapshot cam2;
    uint64_t vector_seq = 0;
    uint64_t cam0_seq = 0;
    uint64_t cam2_seq = 0;
};

void camera_worker(
    SharedState& shared,
    int sensor_id,
    const std::filesystem::path& images_dir)
{
    const std::filesystem::path cam_dir = images_dir / ("cam" + std::to_string(sensor_id));
    std::filesystem::create_directories(cam_dir);

    const std::string pipeline_desc = build_camera_pipeline(sensor_id);
    std::cerr << "pipeline (cam" << sensor_id << "):\n  " << pipeline_desc << "\n";

    GError* err = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeline_desc.c_str(), &err);
    if (!pipeline) {
        std::cerr << "ERROR: gst_parse_launch failed for cam" << sensor_id << "\n";
        if (err) { std::cerr << "  " << err->message << "\n"; g_error_free(err); }
        return;
    }
    if (err) {
        std::cerr << "GStreamer parse warning/error (cam" << sensor_id << "): " << err->message << "\n";
        g_error_free(err);
        err = nullptr;
    }

    GstElement* sink_elem = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!sink_elem) {
        std::cerr << "ERROR: cannot find appsink element 'sink' for cam" << sensor_id << "\n";
        gst_object_unref(pipeline);
        return;
    }
    GstAppSink* appsink = GST_APP_SINK(sink_elem);

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "ERROR: failed to set GStreamer pipeline to PLAYING for cam" << sensor_id << "\n";
        gst_object_unref(sink_elem);
        gst_object_unref(pipeline);
        return;
    }

    uint64_t cam_idx = 0;
    while (true) {
        if (!check_bus_nonblocking(pipeline)) {
            std::cerr << "Stopping camera thread cam" << sensor_id << " due to GStreamer error/EOS.\n";
            break;
        }

        GstSample* sample = gst_app_sink_try_pull_sample(appsink, 100000000ULL);
        if (!sample) continue;

        ++cam_idx;
        char img_name[32];
        std::snprintf(img_name, sizeof(img_name), "%06llu.jpg",
                      static_cast<unsigned long long>(cam_idx));
        const std::filesystem::path img_path = cam_dir / img_name;

        if (!save_jpeg_from_sample(sample, img_path.string().c_str())) {
            std::cerr << "WARN: failed to save camera frame for cam" << sensor_id
                      << " idx=" << cam_idx << "\n";
            gst_sample_unref(sample);
            continue;
        }

        CameraSnapshot cam;
        cam.ts = now_realtime();
        cam.image_path = img_path.string();
        cam.frame_idx = cam_idx;

        {
            std::lock_guard<std::mutex> lk(shared.mtx);
            if (sensor_id == 0) {
                shared.cam0 = std::move(cam);
                ++shared.cam0_seq;
            } else if (sensor_id == 2) {
                shared.cam2 = std::move(cam);
                ++shared.cam2_seq;
            }
        }
        shared.cv.notify_one();

        gst_sample_unref(sample);
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(sink_elem);
    gst_object_unref(pipeline);
}

// -----------------------------------------------------------------------------
// Minimal JSON parser for flat objects:
//   {"key1": <number|"string"|true|false|null>, "key2": <...>, ...}
// Only numbers are stored; non-numeric values become NaN.
// -----------------------------------------------------------------------------
static void skip_ws(const char* s, int n, int& i)
{
    while (i < n) {
        const char c = s[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') ++i;
        else break;
    }
}

static bool parse_json_string(const char* s, int n, int& i, std::string& out)
{
    skip_ws(s, n, i);
    if (i >= n || s[i] != '"') return false;
    ++i;
    out.clear();
    while (i < n) {
        const char c = s[i++];
        if (c == '"') return true;
        if (c == '\\') {
            if (i >= n) return false;
            const char esc = s[i++];
            switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u':
                if (i + 4 > n) return false;
                // Skip unicode escapes (we don't need them for our keys).
                i += 4;
                break;
            default: out.push_back(esc); break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

// Returns true on success. On non-numeric (true/false/null/string) returns true
// with value=NaN. Advances i past the consumed value.
static bool parse_json_value(const char* s, int n, int& i, double& value)
{
    skip_ws(s, n, i);
    if (i >= n) return false;

    const char c = s[i];

    if (c == '"') {
        std::string tmp;
        if (!parse_json_string(s, n, i, tmp)) return false;
        value = std::nan("");
        return true;
    }

    if (c == 't' || c == 'f' || c == 'n') {
        // true/false/null
        const int start = i;
        while (i < n && ((s[i] >= 'a' && s[i] <= 'z'))) ++i;
        const std::string tok(s + start, s + i);
        if (tok == "true") value = 1.0;
        else if (tok == "false") value = 0.0;
        else value = std::nan("");
        return true;
    }

    // number (incl. NaN / Infinity tolerated as written by Python's json with allow_nan)
    const int start = i;
    if (s[i] == '+' || s[i] == '-') ++i;
    while (i < n) {
        const char d = s[i];
        if ((d >= '0' && d <= '9') || d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-' ||
            d == 'N' || d == 'a' || d == 'n' || d == 'I' || d == 'i' || d == 'f' || d == 'y' || d == 't') {
            ++i;
        } else {
            break;
        }
    }
    if (i == start) return false;
    const std::string tok(s + start, s + i);
    try {
        value = std::stod(tok);
    } catch (...) {
        value = std::nan("");
    }
    return true;
}

static bool parse_vector_json(const unsigned char* buf, int n, VectorSnapshot& out)
{
    if (n <= 0) return false;
    const char* s = reinterpret_cast<const char*>(buf);

    int i = 0;
    skip_ws(s, n, i);
    if (i >= n || s[i] != '{') return false;
    ++i;

    out.ts = now_realtime();

    skip_ws(s, n, i);
    if (i < n && s[i] == '}') { ++i; return true; }

    while (i < n) {
        std::string key;
        if (!parse_json_string(s, n, i, key)) return false;
        skip_ws(s, n, i);
        if (i >= n || s[i] != ':') return false;
        ++i;
        double v = 0.0;
        if (!parse_json_value(s, n, i, v)) return false;

        set_vector_signal_value(out, key, v);

        skip_ws(s, n, i);
        if (i < n && s[i] == ',') { ++i; continue; }
        if (i < n && s[i] == '}') { ++i; return true; }
        return false;
    }
    return false;
}

static std::string csv_escape(const std::string& in)
{
    std::string out;
    out.reserve(in.size() + 2);
    out.push_back('"');
    for (const char c : in) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

int main(int argc, char** argv)
{
    // -------- Parse args --------
    Opts opts{};
    if (!parse_args(argc, argv, opts)) return 1;

    // -------- Output folders --------
    const std::filesystem::path log_root_path = opts.log_path.empty()
        ? std::filesystem::path()
        : expand_tilde_path(opts.log_path);
    static std::filesystem::path run_log_dir;
    run_log_dir = create_run_log_dir(log_root_path);
    const std::filesystem::path images_dir = run_log_dir / "images";
    const std::filesystem::path telemetry_csv_path = run_log_dir / "telemetry.csv";
    std::cerr << "Run log directory: " << run_log_dir.string() << "\n";

    // SIGINT handler to print log folder name
    std::signal(SIGINT, [](int){
        std::cout << "\nLog folder: " << run_log_dir.string() << std::endl;
        std::exit(0);
    });

    // -------- CSV log --------
    std::ofstream csv(telemetry_csv_path.string());
    if (!csv) {
        std::cerr << "ERROR: cannot open " << telemetry_csv_path.string() << "\n";
        return 1;
    }
    auto write_csv_header = [&]() {
        csv << "idx,unix_sec,unix_nsec";
        for (const auto& binding : kVectorSignalBindings) {
            csv << "," << binding.json_key;
        }
        csv << "," << kAdmaCsvHeader;
        csv << ",cam0_image,cam2_image\n";
        csv.flush();
    };
    write_csv_header();

    // -------- GStreamer init --------
    gst_init(&argc, &argv);

    SharedState shared{};

    // -------- Vector (UDP/JSON: ETH) thread --------
    std::thread vector_thread([&shared, eth_port = opts.eth_port]() {
        UdpReceiver receiver(eth_port);
        unsigned char buf[65536];
        VectorSnapshot last_snapshot;

        while (true) {
            const int n = receiver.receive(buf, sizeof(buf));
            if (n <= 0) continue;

            VectorSnapshot merged_snapshot = last_snapshot;
            if (!parse_vector_json(buf, n, merged_snapshot)) continue;
            last_snapshot = merged_snapshot;

            {
                std::lock_guard<std::mutex> lk(shared.mtx);
                shared.vector_snapshot = merged_snapshot;
                ++shared.vector_seq;
            }
            shared.cv.notify_one();
        }
    });

    // -------- ADMA (UDP) thread --------
    std::thread adma_thread([&shared]() {
        adma::AdmaPacketDecoder adma_decoder(adma::ProtocolVersion::V334);
        adma::AdmaUdpReceiver adma_receiver(
            "192.168.1.5",
            static_cast<uint16_t>(1211),
            std::optional<std::string>{"192.168.1.50"});

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
                snap.adma_ins_vel_hor_poi3_x = static_cast<double>(adma_packet.insVelHorPOI[2].x) * 0.005;
                snap.adma_ins_vel_hor_poi3_y = static_cast<double>(adma_packet.insVelHorPOI[2].y) * 0.005;
                snap.adma_ins_vel_hor_poi3_z = static_cast<double>(adma_packet.insVelHorPOI[2].z) * 0.005;

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
                snap.adma_acc_body_poi3_x = static_cast<double>(adma_packet.accBodyPOI[2].x) * 0.0004;
                snap.adma_acc_body_poi3_y = static_cast<double>(adma_packet.accBodyPOI[2].y) * 0.0004;
                snap.adma_acc_body_poi3_z = static_cast<double>(adma_packet.accBodyPOI[2].z) * 0.0004;

                snap.adma_acc_horizontal_poi1_x = static_cast<double>(adma_packet.accHorizontalPOI[0].x) * 0.0004;
                snap.adma_acc_horizontal_poi1_y = static_cast<double>(adma_packet.accHorizontalPOI[0].y) * 0.0004;
                snap.adma_acc_horizontal_poi1_z = static_cast<double>(adma_packet.accHorizontalPOI[0].z) * 0.0004;
                snap.adma_acc_horizontal_poi2_x = static_cast<double>(adma_packet.accHorizontalPOI[1].x) * 0.0004;
                snap.adma_acc_horizontal_poi2_y = static_cast<double>(adma_packet.accHorizontalPOI[1].y) * 0.0004;
                snap.adma_acc_horizontal_poi2_z = static_cast<double>(adma_packet.accHorizontalPOI[1].z) * 0.0004;
                snap.adma_acc_horizontal_poi3_x = static_cast<double>(adma_packet.accHorizontalPOI[2].x) * 0.0004;
                snap.adma_acc_horizontal_poi3_y = static_cast<double>(adma_packet.accHorizontalPOI[2].y) * 0.0004;
                snap.adma_acc_horizontal_poi3_z = static_cast<double>(adma_packet.accHorizontalPOI[2].z) * 0.0004;

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
                snap.adma_misc_poi3_side_slip_angle = static_cast<double>(adma_packet.miscPOI[2].sideSlipAngle);
                snap.adma_misc_poi3_distance_traveled = static_cast<double>(adma_packet.miscPOI[2].distanceTraveled);

                snap.adma_ins_pos_lat = static_cast<double>(adma_packet.insPos.pos_abs.latitude);
                snap.adma_ins_pos_lon = static_cast<double>(adma_packet.insPos.pos_abs.longitude);
                snap.adma_ins_height = static_cast<double>(adma_packet.insHeight);

                snap.adma_ins_pos_poi1_lat = static_cast<double>(adma_packet.insPosPOI[0].pos_abs.latitude);
                snap.adma_ins_pos_poi1_lon = static_cast<double>(adma_packet.insPosPOI[0].pos_abs.longitude);
                snap.adma_ins_height_poi1 = static_cast<double>(adma_packet.insHeightPOI[0]);
                snap.adma_ins_pos_poi2_lat = static_cast<double>(adma_packet.insPosPOI[1].pos_abs.latitude);
                snap.adma_ins_pos_poi2_lon = static_cast<double>(adma_packet.insPosPOI[1].pos_abs.longitude);
                snap.adma_ins_height_poi2 = static_cast<double>(adma_packet.insHeightPOI[1]);
                snap.adma_ins_pos_poi3_lat = static_cast<double>(adma_packet.insPosPOI[2].pos_abs.latitude);
                snap.adma_ins_pos_poi3_lon = static_cast<double>(adma_packet.insPosPOI[2].pos_abs.longitude);
                snap.adma_ins_height_poi3 = static_cast<double>(adma_packet.insHeightPOI[2]);

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

    // -------- Camera threads --------
    std::thread camera0_thread(camera_worker, std::ref(shared), 0, std::cref(images_dir));
    std::thread camera2_thread(camera_worker, std::ref(shared), 2, std::cref(images_dir));

    uint64_t idx = 0;
    uint64_t consumed_vector_seq = 0;
    uint64_t consumed_cam0_seq = 0;
    uint64_t consumed_cam2_seq = 0;

    while (true) {
        VectorSnapshot vector_snapshot;
        AdmaSnapshot adma_snapshot;
        CameraSnapshot cam0;
        CameraSnapshot cam2;

        {
            std::unique_lock<std::mutex> lk(shared.mtx);
            shared.cv.wait(lk, [&]() {
                return shared.vector_seq != consumed_vector_seq ||
                       shared.cam0_seq != consumed_cam0_seq ||
                       shared.cam2_seq != consumed_cam2_seq;
            });
            consumed_vector_seq = shared.vector_seq;
            consumed_cam0_seq = shared.cam0_seq;
            consumed_cam2_seq = shared.cam2_seq;
            vector_snapshot = shared.vector_snapshot;
            adma_snapshot = shared.adma;
            cam0 = shared.cam0;
            cam2 = shared.cam2;
        }

        if (cam0.image_path.empty() && cam2.image_path.empty()) {
            continue;
        }

        if (vector_snapshot.ts.tv_sec == 0 && vector_snapshot.ts.tv_nsec == 0) {
            vector_snapshot.ts = now_realtime();
        }

        ++idx;

        print_line(adma_snapshot.adma_kf_status,
                   adma_snapshot.adma_kf_lat_stimulated,
                   adma_snapshot.adma_kf_long_stimulated,
                   adma_snapshot.adma_kf_steady_state,
                   vector_snapshot.LWI_AgStgWhl,
                   vector_snapshot.VDSO_Vx3dKmph,
                   vector_snapshot.ts);

        csv << idx << ","
            << vector_snapshot.ts.tv_sec << "," << vector_snapshot.ts.tv_nsec;
        for (const auto& binding : kVectorSignalBindings) {
            csv << "," << vector_snapshot.*(binding.member);
        }
        csv << ",";
        write_adma_csv_values(csv, adma_snapshot);
        csv << "," << csv_escape(cam0.image_path)
            << "," << csv_escape(cam2.image_path)
            << "\n";
        csv.flush();
    }

    vector_thread.join();
    adma_thread.join();
    camera0_thread.join();
    camera2_thread.join();
    return 0;
}
