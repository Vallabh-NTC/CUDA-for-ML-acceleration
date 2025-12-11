#include "UdpReceiver.hpp"
#include "LWI01.hpp"
#include "Lichthinten01.hpp"
#include "ChassisState.hpp"
#include "CudaMemAssign.hpp"
#include "SARA.hpp"
#include "BrakeEV01.hpp"
#include "Motor20.hpp"
#include "ESP21.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <clocale>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

// GPU launcher function
extern "C" void launch_kernel(ChassisState* ring, int index);

// PCAP header structures
#pragma pack(push, 1)
struct PcapGlobalHeader {
    uint32_t magic_number;   // 0xa1b2c3d4
    uint16_t version_major;  // 2
    uint16_t version_minor;  // 4
    int32_t  thiszone;       // GMT to local correction, usually 0
    uint32_t sigfigs;        // accuracy of timestamps, usually 0
    uint32_t snaplen;        // max length of captured packets, in octets
    uint32_t network;        // data link type (USER0 = 147)
};

struct PcapRecordHeader {
    uint32_t ts_sec;   // timestamp seconds (UNIX time)
    uint32_t ts_usec;  // timestamp microseconds
    uint32_t incl_len; // number of octets of packet saved in file
    uint32_t orig_len; // actual length of packet
};
#pragma pack(pop)

int main()
{
    // Force dot as decimal separator
    std::setlocale(LC_NUMERIC, "C");

    using Clock = std::chrono::high_resolution_clock;

    CudaMemAssign mem(64);
    int writeIndex = 0;

    UdpReceiver receiver(1500);

    unsigned char buf[4096];

    // ---------- Build PCAP filename with timestamp ----------
    // Format: log_YYYYMMDD_HHMMSS.pcap, saved in /home/nvidia (current dir)
    std::time_t now = std::time(nullptr);
    std::tm tm_now{};
    localtime_r(&now, &tm_now);   // Xavier is Linux, so localtime_r is available

    char tsbuf[32];
    std::strftime(tsbuf, sizeof(tsbuf), "%Y%m%d_%H%M%S", &tm_now);
    std::string pcapFilename = std::string("log_") + tsbuf + ".pcap";

    std::cout << "Start decoder + GPU pipeline + PCAP logging\n";
    std::cout << "PCAP file: " << pcapFilename << "\n";

    // ---------- Open PCAP file ----------
    std::ofstream pcapFile(pcapFilename, std::ios::binary);
    if (!pcapFile) {
        std::cerr << "Error: cannot open " << pcapFilename << "\n";
        return 1;
    }

    // Write global PCAP header
    PcapGlobalHeader gh{};
    gh.magic_number  = 0xa1b2c3d4;
    gh.version_major = 2;
    gh.version_minor = 4;
    gh.thiszone      = 0;
    gh.sigfigs       = 0;
    gh.snaplen       = 65535;
    // LINKTYPE_USER0 (147): user-defined data
    gh.network       = 147;

    pcapFile.write(reinterpret_cast<const char*>(&gh), sizeof(gh));

    // ---------- Main loop ----------
    while (true)
    {
        auto loop_start = Clock::now();

        // ---- Receive UDP ----
        int n = receiver.receive(buf, sizeof(buf));
        if (n <= 0) continue;
        if (n < 700) continue;
        if (buf[0] != 1) continue;

        // ---- Timestamp for PCAP + log ----
        timespec ts_pcap{};
        clock_gettime(CLOCK_REALTIME, &ts_pcap);

        // ---- Write raw UDP packet to PCAP ----
        PcapRecordHeader rh{};
        rh.ts_sec   = static_cast<uint32_t>(ts_pcap.tv_sec);
        rh.ts_usec  = static_cast<uint32_t>(ts_pcap.tv_nsec / 1000);
        rh.incl_len = static_cast<uint32_t>(n);
        rh.orig_len = static_cast<uint32_t>(n);

        pcapFile.write(reinterpret_cast<const char*>(&rh), sizeof(rh));
        pcapFile.write(reinterpret_cast<const char*>(buf), n);
        // For performance you can rely on buffering and not flush every time
        // pcapFile.flush();

        const unsigned char* pdus = buf + 3;

        // ---- Decode signals ----
        LWI01 lwi;
        lwi.decode(pdus + 677);

        Lichthinten01 lh;
        lh.decode(pdus + 605);

        SARA sara;
        sara.decode_all(pdus);

        BrakeEV01 brk;
        brk.decode(pdus + 364);   // your offset

        Motor20 m20;
        m20.decode(pdus + 629);
        float gas = m20.data().gas_percent;

        ESP21 esp;
        esp.decode(pdus + 542);
        float veh_speed = esp.data().vehicle_speed; // m/s (as in your decoder)

        // ---- Write into ring buffer slot ----
        ChassisState& slot = mem.host_ring[writeIndex];
        slot.steering_angle = lwi.angle;
        slot.steering_speed = lwi.speed;
        slot.lights_rear    = lh.compute_mask();

        timespec ts{};
        clock_gettime(CLOCK_REALTIME, &ts);
        slot.timestamp_ns =
            uint64_t(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;

        int processedIndex = writeIndex;
        writeIndex = (writeIndex + 1) % mem.capacity;

        // ---- Launch GPU kernel ----
        launch_kernel(mem.device_ring, processedIndex);

        // ---- End of loop: compute processing time in milliseconds ----
        auto loop_end = Clock::now();
        double loop_ms =
            std::chrono::duration<double, std::milli>(loop_end - loop_start).count();

        // ---- Format human-readable timestamp from ts_pcap ----
        char time_buf[32];
        std::tm tm_local{};
        localtime_r(&ts_pcap.tv_sec, &tm_local);   // local time on Xavier

        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm_local);
        long ms = ts_pcap.tv_nsec / 1'000'000;     // nanoseconds -> milliseconds

        // ---- Print all signals to terminal with timestamp ----
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[" << time_buf << "." << std::setw(3) << std::setfill('0') << ms
                  << std::setfill(' ') << "] "
                  << "Index=" << processedIndex
                  << " | steering_angle=" << slot.steering_angle
                  << " deg, steering_speed=" << slot.steering_speed
                  << " deg/s, accel_x=" << sara.d10.accel_x
                  << " m/s^2, accel_y=" << sara.d10.accel_y
                  << " m/s^2, accel_z=" << sara.d08.accel_z
                  << " m/s^2, omega_x=" << sara.d08.omega_x
                  << " rad/s, omega_y=" << sara.d08.omega_y
                  << " rad/s, omega_z=" << sara.d10.omega_z
                  << " rad/s, nickwinkel=" << sara.d07.nickwinkel
                  << " deg, wankwinkel=" << sara.d07.wankwinkel
                  << " deg, brake=" << brk.brake_percent
                  << " %, gas=" << gas
                  << " %, veh_speed=" << veh_speed
                  << " m/s"
                  << " | Loop time=" << loop_ms << " ms\n";
    }

    // (practically never reached)
    // pcapFile.close();
    return 0;
}
