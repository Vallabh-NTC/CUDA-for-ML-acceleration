#include "UdpReceiver.hpp"
#include "LWI01.hpp"
#include "ChassisState.hpp"
#include "CudaMemAssign.hpp"
#include "SARA_10.hpp"
#include "SARA_08.hpp"
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
#include <cstring>
#include <array>

// GPU launcher function
extern "C" void launch_kernel(ChassisState* ring, int index);

// ------------------------------------------------------------------
// Logging thread globals
// ------------------------------------------------------------------
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

            std::cout << msg << std::endl; // slow terminal I/O here
            lock.lock();
        }
    }
}

// ------------------------------------------------------------------
// PCAP header structures
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// Packet queue between RX thread and worker thread
// ------------------------------------------------------------------
struct Packet
{
    int len = 0;
    unsigned char data[4096];
    timespec ts{};          // timestamp di arrivo
    uint64_t timestamp_ms;  // in ms dall'epoch (per dt_ms/freq)
};

std::mutex pktMutex;
std::condition_variable pktCv;
std::queue<Packet> pktQueue;
bool runRxWorker = true;

// ------------------------------------------------------------------
// RX thread: mantiene il rate a 5 ms, fa solo recv + timestamp + push
// ------------------------------------------------------------------
void rxThreadFunc(UdpReceiver& receiver)
{
    unsigned char buf[4096];

    while (runRxWorker) {
        int n = receiver.receive(buf, sizeof(buf));
        if (n <= 0) {
            continue;
        }

        timespec ts{};
        clock_gettime(CLOCK_REALTIME, &ts);

        uint64_t ms =
            uint64_t(ts.tv_sec) * 1000ULL +
            uint64_t(ts.tv_nsec / 1'000'000ULL);

        Packet pkt{};
        pkt.len = n;
        pkt.ts = ts;
        pkt.timestamp_ms = ms;
        std::memcpy(pkt.data, buf, n);

        {
            std::lock_guard<std::mutex> lock(pktMutex);
            pktQueue.push(std::move(pkt));
        }
        pktCv.notify_one();
    }
}

// ------------------------------------------------------------------
// Worker thread: PCAP + decode + GPU + log
// ------------------------------------------------------------------
void workerThreadFunc(CudaMemAssign& mem, std::ofstream& pcapFile)
{
    using Clock = std::chrono::high_resolution_clock;

    uint64_t prev_ts_ms = 0;
    bool have_prev_ts   = false;
    uint64_t pktCounter = 0;
    int writeIndex      = 0;

    // ---- Unique offsets (as agreed / extracted) ----
    static constexpr std::array<int, 32> lwi01_offsets = {{
        32, 56, 83, 96, 151, 162, 172, 185, 192, 221, 242, 250, 305, 348, 357, 389,
        416, 418, 470, 511, 575, 625, 637, 653, 677, 678, 715, 802, 818, 1027, 1061, 1142
    }};

    static constexpr std::array<int, 61> sara10_offsets = {{
        0, 16, 27, 48, 64, 80, 95, 111, 112, 118, 128, 130, 151, 170, 189, 213,
        229, 242, 259, 267, 274, 289, 303, 318, 321, 326, 331, 340, 353, 366,
        367, 373, 380, 385, 402, 408, 412, 420, 436, 457, 468, 489, 518, 520,
        522, 528, 589, 595, 609, 617, 620, 652, 671, 682, 684, 692, 747, 792,
        856, 1187, 1202
    }};

    static constexpr std::array<int, 60> sara08_offsets = {{
        0, 25, 32, 45, 70, 73, 77, 86, 108, 110, 121, 125, 150, 153, 158, 161,
        192, 206, 230, 241, 252, 260, 261, 273, 288, 307, 348, 349, 369, 374,
        378, 385, 389, 403, 418, 425, 437, 464, 468, 518, 553, 560, 561, 603,
        614, 654, 656, 661, 680, 698, 759, 813, 820, 838, 849, 855, 866, 1062,
        1114, 1130
    }};

    static constexpr std::array<int, 30> brkev01_offsets = {{
        16, 40, 48, 57, 116, 172, 177, 196, 248, 254, 261, 284, 364, 372, 381,
        392, 422, 454, 536, 537, 544, 582, 593, 661, 684, 728, 894, 958, 1029, 1235
    }};

    static constexpr std::array<int, 16> esp21_offsets = {{
        0, 74, 150, 196, 202, 270, 340, 341, 375, 434, 480, 542, 628, 659, 1106, 1179
    }};

    static constexpr std::array<int, 31> motor20_offsets = {{
        8, 65, 68, 78, 79, 117, 170, 179, 204, 213, 233, 349, 359, 365, 374, 434,
        446, 497, 559, 598, 612, 617, 620, 629, 689, 841, 964, 1003, 1064, 1098, 1158
    }};

    while (true) {
        // ----- Wait for packet from RX thread -----
        Packet pkt;
        {
            std::unique_lock<std::mutex> lock(pktMutex);
            pktCv.wait(lock, [] {
                return !pktQueue.empty() || !runRxWorker;
            });

            if (!runRxWorker && pktQueue.empty()) {
                break;
            }

            pkt = std::move(pktQueue.front());
            pktQueue.pop();
        }

        ++pktCounter;

        auto loop_start = Clock::now();

        // ----- dt_ms & freq sui PACCHETTI ARRIVATI -----
        double dt_ms   = 0.0;
        double freq_hz = 0.0;
        if (have_prev_ts) {
            uint64_t diff = pkt.timestamp_ms - prev_ts_ms;
            dt_ms = static_cast<double>(diff);
            if (dt_ms > 0.0) {
                freq_hz = 1000.0 / dt_ms;
            }
        }
        prev_ts_ms = pkt.timestamp_ms;
        have_prev_ts = true;

        // ----- PCAP write -----
        PcapRecordHeader rh{};
        rh.ts_sec   = pkt.ts.tv_sec;
        rh.ts_usec  = pkt.ts.tv_nsec / 1000;
        rh.incl_len = pkt.len;
        rh.orig_len = pkt.len;

        pcapFile.write(reinterpret_cast<const char*>(&rh), sizeof(rh));
        pcapFile.write(reinterpret_cast<const char*>(pkt.data), pkt.len);

        // ----- Basic checks -----
        //int n = pkt.len;
        const unsigned char* buf = pkt.data;

        // bool valid_for_decode = true;
        // std::string reason;

        // if (n < 700) {
        //     valid_for_decode = false;
        //     reason = "len<700";
        // } else if (buf[0] != 1) {
        //     valid_for_decode = false;
        //     reason = "buf[0]!=1";
        // }

        // if (!valid_for_decode) {
        //     std::ostringstream ss;
        //     ss << "[PKT " << pktCounter << "] "
        //        << "IGNORED (" << reason << ") "
        //        << "n=" << n
        //        << ", buf[0]=" << int(static_cast<unsigned char>(buf[0]))
        //        << ", t_ms=" << pkt.timestamp_ms
        //        << ", dt_ms=" << dt_ms
        //        << ", freq_Hz=" << freq_hz;

        //     {
        //         std::lock_guard<std::mutex> lock(logMutex);
        //         logQueue.emplace(ss.str());
        //     }
        //     logCv.notify_one();
        //     continue;
        // }

        // ----- Decode PDUs -----
        const unsigned char* pdus = buf + 3;

        // LWI01
        std::array<LWI01, lwi01_offsets.size()> lwi_all;
        for (std::size_t i = 0; i < lwi01_offsets.size(); ++i) {
            lwi_all[i].decode(pdus + lwi01_offsets[i]);
        }
        const LWI01* chosen_lwi = nullptr;
        for (std::size_t i = 0; i < lwi01_offsets.size(); ++i) {
            if (lwi01_offsets[i] == 677) {
                chosen_lwi = &lwi_all[i];
                break;
            }
        }
        if (!chosen_lwi) chosen_lwi = &lwi_all[0];

        // SARA10
        std::array<SARA_10, sara10_offsets.size()> sara10_all;
        for (std::size_t i = 0; i < sara10_offsets.size(); ++i) {
            sara10_all[i].decode(pdus + sara10_offsets[i]);
        }
        const SARA_10_Data* chosen_sara10 = nullptr;
        for (std::size_t i = 0; i < sara10_offsets.size(); ++i) {
            if (sara10_offsets[i] == 412) {
                chosen_sara10 = &sara10_all[i].data();
                break;
            }
        }
        if (!chosen_sara10) chosen_sara10 = &sara10_all[0].data();

        // SARA08
        std::array<SARA_08, sara08_offsets.size()> sara08_all;
        for (std::size_t i = 0; i < sara08_offsets.size(); ++i) {
            sara08_all[i].decode(pdus + sara08_offsets[i]);
        }
        const SARA_08_Data* chosen_sara08 = nullptr;
        for (std::size_t i = 0; i < sara08_offsets.size(); ++i) {
            if (sara08_offsets[i] == 437) {
                chosen_sara08 = &sara08_all[i].data();
                break;
            }
        }
        if (!chosen_sara08) chosen_sara08 = &sara08_all[0].data();

        // BrakeEV01
        std::array<BrakeEV01, brkev01_offsets.size()> brk_all;
        for (std::size_t i = 0; i < brkev01_offsets.size(); ++i) {
            brk_all[i].decode(pdus + brkev01_offsets[i]);
        }
        const BrakeEV01* chosen_brk = nullptr;
        for (std::size_t i = 0; i < brkev01_offsets.size(); ++i) {
            if (brkev01_offsets[i] == 364) {
                chosen_brk = &brk_all[i];
                break;
            }
        }
        if (!chosen_brk) chosen_brk = &brk_all[0];

        // ESP21
        std::array<ESP21, esp21_offsets.size()> esp_all;
        for (std::size_t i = 0; i < esp21_offsets.size(); ++i) {
            esp_all[i].decode(pdus + esp21_offsets[i]);
        }
        const ESP21* chosen_esp = nullptr;
        for (std::size_t i = 0; i < esp21_offsets.size(); ++i) {
            if (esp21_offsets[i] == 542) {
                chosen_esp = &esp_all[i];
                break;
            }
        }
        if (!chosen_esp) chosen_esp = &esp_all[0];

        // Motor20
        std::array<Motor20, motor20_offsets.size()> motor20_all;
        for (std::size_t i = 0; i < motor20_offsets.size(); ++i) {
            motor20_all[i].decode(pdus + motor20_offsets[i]);
        }
        const Motor20* chosen_m20 = nullptr;
        for (std::size_t i = 0; i < motor20_offsets.size(); ++i) {
            if (motor20_offsets[i] == 629) {
                chosen_m20 = &motor20_all[i];
                break;
            }
        }
        if (!chosen_m20) chosen_m20 = &motor20_all[0];

        float gas = chosen_m20->data().gas_percent;
        float veh_speed = chosen_esp->data().vehicle_speed;

        // ----- Write into ring buffer -----
        ChassisState& slot = mem.host_ring[writeIndex];
        slot.steering_angle = chosen_lwi->angle;
        slot.steering_speed = chosen_lwi->speed;
        slot.timestamp_ms   = pkt.timestamp_ms;

        int processedIndex = writeIndex;
        writeIndex = (writeIndex + 1) % mem.capacity;

        // ----- GPU kernel -----
        launch_kernel(mem.device_ring, processedIndex);

        // ----- loop time -----
        auto loop_end = Clock::now();
        double loop_ms =
            std::chrono::duration<double, std::milli>(loop_end - loop_start).count();

        // ----- Log -----
        char time_buf[32];
        std::tm tm_local{};
        localtime_r(&pkt.ts.tv_sec, &tm_local);
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm_local);
        long ms = pkt.ts.tv_nsec / 1'000'000;

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss << "[PKT " << pktCounter << "] "
           << "[" << time_buf << "." << std::setw(3) << std::setfill('0') << ms
           << std::setfill(' ') << "] "
           << "DECODED n=" << n
           << ", dt_ms=" << dt_ms
           << ", freq_Hz=" << freq_hz
           << " | Index=" << processedIndex
           << " | steer=" << slot.steering_angle
           << ", steer_spd=" << slot.steering_speed
           << ", ax=" << chosen_sara10->accel_x
           << ", ay=" << chosen_sara10->accel_y
           << ", az=" << chosen_sara08->accel_z
           << ", ox=" << chosen_sara08->omega_x
           << ", oy=" << chosen_sara08->omega_y
           << ", oz=" << chosen_sara10->omega_z
           << ", brake=" << chosen_brk->brake_percent
           << ", gas=" << gas
           << ", v=" << veh_speed
           << " | loop=" << loop_ms << " ms";

        {
            std::lock_guard<std::mutex> lock(logMutex);
            logQueue.emplace(ss.str());
        }
        logCv.notify_one();
    }
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------
int main()
{
    std::setlocale(LC_NUMERIC, "C");

    CudaMemAssign mem(64);
    UdpReceiver receiver(1500);

    std::thread logger(loggingThreadFunc);

    std::time_t now = std::time(nullptr);
    std::tm tm_now{};
    localtime_r(&now, &tm_now);

    char tsbuf[32];
    std::strftime(tsbuf, sizeof(tsbuf), "%Y%m%d_%H%M%S", &tm_now);
    std::string pcapFilename = std::string("log_") + tsbuf + ".pcap";

    {
        std::ostringstream ss;
        ss << "Start decoder + GPU pipeline + PCAP logging (2-thread RX/worker)\n"
           << "PCAP file: " << pcapFilename;
        std::lock_guard<std::mutex> lock(logMutex);
        logQueue.emplace(ss.str());
        logCv.notify_one();
    }

    std::ofstream pcapFile(pcapFilename, std::ios::binary);
    if (!pcapFile) {
        std::cerr << "Error: cannot open " << pcapFilename << "\n";
        runLogger = false;
        logCv.notify_one();
        logger.join();
        return 1;
    }

    PcapGlobalHeader gh{};
    gh.magic_number  = 0xa1b2c3d4;
    gh.version_major = 2;
    gh.version_minor = 4;
    gh.thiszone      = 0;
    gh.sigfigs       = 0;
    gh.snaplen       = 65535;
    gh.network       = 147;

    pcapFile.write(reinterpret_cast<const char*>(&gh), sizeof(gh));

    std::thread rxThread(rxThreadFunc, std::ref(receiver));
    std::thread workerThread(workerThreadFunc, std::ref(mem), std::ref(pcapFile));

    rxThread.join();
    workerThread.join();

    runLogger = false;
    logCv.notify_one();
    logger.join();

    return 0;
}
