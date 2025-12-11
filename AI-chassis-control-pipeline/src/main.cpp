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

#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <sstream>

// GPU launcher function
extern "C" void launch_kernel(ChassisState* ring, int index);

// -------------- Logging thread globals --------------
std::mutex logMutex;
std::condition_variable logCv;
std::queue<std::string> logQueue;
bool runLogger = true;

// Logging thread
void loggingThreadFunc()
{
    while (runLogger) {
        std::unique_lock<std::mutex> lock(logMutex);
        logCv.wait(lock, [] { return !logQueue.empty() || !runLogger; });

        while (!logQueue.empty()) {
            std::string msg = std::move(logQueue.front());
            logQueue.pop();
            lock.unlock();

            std::cout << msg << std::endl; // slow terminal I/O done here
            lock.lock();
        }
    }
}

// PCAP header structures
#pragma pack(push, 1)
struct PcapGlobalHeader {
    uint32_t magic_number;   // 0xa1b2c3d4
    uint16_t version_major;  // 2
    uint16_t version_minor;  // 4
    int32_t  thiszone;
    uint32_t sigfigs;
    uint32_t snaplen;
    uint32_t network;        // USER0 = 147
};

struct PcapRecordHeader {
    uint32_t ts_sec;
    uint32_t ts_usec;
    uint32_t incl_len;
    uint32_t orig_len;
};
#pragma pack(pop)

int main()
{
    std::setlocale(LC_NUMERIC, "C");

    using Clock = std::chrono::high_resolution_clock;

    CudaMemAssign mem(64);
    int writeIndex = 0;
    UdpReceiver receiver(1500);

    unsigned char buf[4096];

    // ---------- Start logging thread ----------
    std::thread logger(loggingThreadFunc);

    // ---------- Build PCAP filename ----------
    std::time_t now = std::time(nullptr);
    std::tm tm_now{};
    localtime_r(&now, &tm_now);

    char tsbuf[32];
    std::strftime(tsbuf, sizeof(tsbuf), "%Y%m%d_%H%M%S", &tm_now);
    std::string pcapFilename = std::string("log_") + tsbuf + ".pcap";

    {
        std::ostringstream ss;
        ss << "Start decoder + GPU pipeline + PCAP logging\nPCAP file: " << pcapFilename;
        std::lock_guard<std::mutex> lock(logMutex);
        logQueue.emplace(ss.str());
        logCv.notify_one();
    }

    // ---------- Open PCAP file ----------
    std::ofstream pcapFile(pcapFilename, std::ios::binary);
    if (!pcapFile) {
        std::cerr << "Error: cannot open " << pcapFilename << "\n";
        runLogger = false;
        logCv.notify_one();
        logger.join();
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

        // ---- Timestamp for PCAP ----
        timespec ts_pcap{};
        clock_gettime(CLOCK_REALTIME, &ts_pcap);

        // ---- Write to PCAP ----
        PcapRecordHeader rh{};
        rh.ts_sec   = ts_pcap.tv_sec;
        rh.ts_usec  = ts_pcap.tv_nsec / 1000;
        rh.incl_len = n;
        rh.orig_len = n;

        pcapFile.write(reinterpret_cast<const char*>(&rh), sizeof(rh));
        pcapFile.write(reinterpret_cast<const char*>(buf), n);

        const unsigned char* pdus = buf + 3;

        // ---- Decode signals ----
        LWI01 lwi;
        lwi.decode(pdus + 677);

        Lichthinten01 lh;
        lh.decode(pdus + 605);

        SARA sara;
        sara.decode_all(pdus);

        BrakeEV01 brk;
        brk.decode(pdus + 364);

        Motor20 m20;
        m20.decode(pdus + 629);
        float gas = m20.data().gas_percent;

        ESP21 esp;
        esp.decode(pdus + 542);
        float veh_speed = esp.data().vehicle_speed;

        // ---- Write into ring buffer ----
        ChassisState& slot = mem.host_ring[writeIndex];
        slot.steering_angle = lwi.angle;
        slot.steering_speed = lwi.speed;
        slot.lights_rear    = lh.compute_mask();

        timespec ts{};
        clock_gettime(CLOCK_REALTIME, &ts);
        slot.timestamp_ns = uint64_t(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;

        int processedIndex = writeIndex;
        writeIndex = (writeIndex + 1) % mem.capacity;

        // ---- Launch GPU kernel ----
        launch_kernel(mem.device_ring, processedIndex);

        // ---- Compute loop time ----
        auto loop_end = Clock::now();
        double loop_ms =
            std::chrono::duration<double, std::milli>(loop_end - loop_start).count();

        // ---- Prepare log message (fast string building, NO cout) ----
        char time_buf[32];
        std::tm tm_local{};
        localtime_r(&ts_pcap.tv_sec, &tm_local);
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm_local);
        long ms = ts_pcap.tv_nsec / 1'000'000;

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss << "[" << time_buf << "." << std::setw(3) << std::setfill('0') << ms
           << std::setfill(' ') << "] "
           << "Index=" << processedIndex
           << " | steering_angle=" << slot.steering_angle
           << ", steering_speed=" << slot.steering_speed
           << ", accel_x=" << sara.d10.accel_x
           << ", accel_y=" << sara.d10.accel_y
           << ", accel_z=" << sara.d08.accel_z
           << ", omega_x=" << sara.d08.omega_x
           << ", omega_y=" << sara.d08.omega_y
           << ", omega_z=" << sara.d10.omega_z
           << ", nickwinkel=" << sara.d07.nickwinkel
           << ", wankwinkel=" << sara.d07.wankwinkel
           << ", brake=" << brk.brake_percent
           << ", gas=" << gas
           << ", veh_speed=" << veh_speed
           << " | Loop time=" << loop_ms << " ms";

        // Push into queue
        {
            std::lock_guard<std::mutex> lock(logMutex);
            logQueue.emplace(ss.str());
        }
        logCv.notify_one();
    }

    // Cleanup (never reached normally)
    runLogger = false;
    logCv.notify_one();
    logger.join();

    return 0;
}
