#include "UdpReceiver.hpp"
#include "LWI01.hpp"
#include "Lichthinten01.hpp"
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
            // timeout/errore -> salta
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

    // ---- Unique offsets (as agreed) ----
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

    while (true) {
        // ----- Wait for packet from RX thread -----
        Packet pkt;
        {
            std::unique_lock<std::mutex> lock(pktMutex);
            pktCv.wait(lock, [] {
                return !pktQueue.empty() || !runRxWorker;
            });

            if (!runRxWorker && pktQueue.empty()) {
                break;  // shutdown
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

        // ----- PCAP write (tutti i pacchetti) -----
        PcapRecordHeader rh{};
        rh.ts_sec   = pkt.ts.tv_sec;
        rh.ts_usec  = pkt.ts.tv_nsec / 1000;
        rh.incl_len = pkt.len;
        rh.orig_len = pkt.len;

        pcapFile.write(reinterpret_cast<const char*>(&rh), sizeof(rh));
        pcapFile.write(reinterpret_cast<const char*>(pkt.data), pkt.len);

        // ----- Controlli base -----
        int n = pkt.len;
        const unsigned char* buf = pkt.data;

        bool valid_for_decode = true;
        std::string reason;

        if (n < 700) {
            valid_for_decode = false;
            reason = "len<700";
        } else if (buf[0] != 1) {
            valid_for_decode = false;
            reason = "buf[0]!=1";
        }

        // ----- Log semplice per pacchetti ignorati -----
        if (!valid_for_decode) {
            std::ostringstream ss;
            ss << "[PKT " << pktCounter << "] "
               << "IGNORED (" << reason << ") "
               << "n=" << n
               << ", buf[0]=" << int(static_cast<unsigned char>(buf[0]))
               << ", t_ms=" << pkt.timestamp_ms
               << ", dt_ms=" << dt_ms
               << ", freq_Hz=" << freq_hz;

            {
                std::lock_guard<std::mutex> lock(logMutex);
                logQueue.emplace(ss.str());
            }
            logCv.notify_one();

            continue;
        }

        // ----- Decode PDUs -----
        const unsigned char* pdus = buf + 3;

        LWI01 lwi;
        lwi.decode(pdus + 677);

        Lichthinten01 lh;
        lh.decode(pdus + 605);

        // ---- Decode ALL unique SARA10 blocks ----
        std::array<SARA_10, sara10_offsets.size()> sara10_all;
        for (std::size_t i = 0; i < sara10_offsets.size(); ++i) {
            sara10_all[i].decode(pdus + sara10_offsets[i]);
        }

        // ---- Decode ALL unique SARA08 blocks ----
        std::array<SARA_08, sara08_offsets.size()> sara08_all;
        for (std::size_t i = 0; i < sara08_offsets.size(); ++i) {
            sara08_all[i].decode(pdus + sara08_offsets[i]);
        }

        // Choose which instance to use for logging/processing:
        // - SARA10: prefer offset 412 (matches your old fixed-offset decode)
        // - SARA08: prefer offset 437 (matches your old fixed-offset decode)
        const SARA_10_Data* chosen_sara10 = nullptr;
        for (std::size_t i = 0; i < sara10_offsets.size(); ++i) {
            if (sara10_offsets[i] == 412) {
                chosen_sara10 = &sara10_all[i].data();
                break;
            }
        }
        if (!chosen_sara10) chosen_sara10 = &sara10_all[0].data();

        const SARA_08_Data* chosen_sara08 = nullptr;
        for (std::size_t i = 0; i < sara08_offsets.size(); ++i) {
            if (sara08_offsets[i] == 437) {
                chosen_sara08 = &sara08_all[i].data();
                break;
            }
        }
        if (!chosen_sara08) chosen_sara08 = &sara08_all[0].data();

        BrakeEV01 brk;
        brk.decode(pdus + 364);

        Motor20 m20;
        m20.decode(pdus + 629);
        float gas = m20.data().gas_percent;

        ESP21 esp;
        esp.decode(pdus + 542);
        float veh_speed = esp.data().vehicle_speed;

        // ----- Write into ring buffer -----
        ChassisState& slot = mem.host_ring[writeIndex];
        slot.steering_angle = lwi.angle;
        slot.steering_speed = lwi.speed;
        slot.lights_rear    = lh.compute_mask();
        slot.timestamp_ms   = pkt.timestamp_ms;

        int processedIndex = writeIndex;
        writeIndex = (writeIndex + 1) % mem.capacity;

        // ----- GPU kernel -----
        launch_kernel(mem.device_ring, processedIndex);

        // ----- Compute worker loop time -----
        auto loop_end = Clock::now();
        double loop_ms =
            std::chrono::duration<double, std::milli>(loop_end - loop_start).count();

        // ----- Build rich log message -----
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
           << ", buf[0]=" << int(static_cast<unsigned char>(buf[0]))
           << " | t_ms=" << pkt.timestamp_ms
           << ", dt_ms=" << dt_ms
           << ", freq_Hz=" << freq_hz
           << " | Index=" << processedIndex
           << " | steering_angle=" << slot.steering_angle
           << ", steering_speed=" << slot.steering_speed
           << ", accel_x=" << chosen_sara10->accel_x
           << ", accel_y=" << chosen_sara10->accel_y
           << ", accel_z=" << chosen_sara08->accel_z
           << ", omega_x=" << chosen_sara08->omega_x
           << ", omega_y=" << chosen_sara08->omega_y
           << ", omega_z=" << chosen_sara10->omega_z
           << ", brake=" << brk.brake_percent
           << ", gas=" << gas
           << ", veh_speed=" << veh_speed
           << " | Worker loop time=" << loop_ms << " ms";

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

    // ---------- CUDA ring buffer ----------
    CudaMemAssign mem(64);

    // ---------- UDP receiver ----------
    UdpReceiver receiver(1500);

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
        ss << "Start decoder + GPU pipeline + PCAP logging (2-thread RX/worker)\n"
           << "PCAP file: " << pcapFilename;
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

    // Global PCAP header
    PcapGlobalHeader gh{};
    gh.magic_number  = 0xa1b2c3d4;
    gh.version_major = 2;
    gh.version_minor = 4;
    gh.thiszone      = 0;
    gh.sigfigs       = 0;
    gh.snaplen       = 65535;
    gh.network       = 147;

    pcapFile.write(reinterpret_cast<const char*>(&gh), sizeof(gh));

    // ---------- Start RX + worker threads ----------
    std::thread rxThread(rxThreadFunc, std::ref(receiver));
    std::thread workerThread(workerThreadFunc, std::ref(mem), std::ref(pcapFile));

    // Per ora non gestiamo shutdown elegante: Ctrl+C e basta.
    rxThread.join();
    workerThread.join();

    runLogger = false;
    logCv.notify_one();
    logger.join();

    return 0;
}
