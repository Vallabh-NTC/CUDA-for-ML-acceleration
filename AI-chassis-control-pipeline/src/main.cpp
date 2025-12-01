#include "UdpReceiver.hpp"
#include "UdpSender.hpp"
#include "LWI01.hpp"
#include "Lichthinten01.hpp"
#include "ChassisState.hpp"
#include "CudaMemAssign.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <clocale>   // per forzare locale "C"

// GPU launcher function
extern "C" void launch_kernel(ChassisState* ring, int index);

int main()
{
    // ---- Forza il punto come separatore decimale ----
    std::setlocale(LC_NUMERIC, "C");

    CudaMemAssign mem(64);
    int writeIndex = 0;

    UdpReceiver receiver(1500);
    UdpSender sender("192.168.1.180", 1600);  // Windows IP

    unsigned char buf[4096];

    std::cout << "Start decoder + GPU pipeline\n";

    using Clock = std::chrono::high_resolution_clock;

    while (true)
    {
        auto loop_start = Clock::now();

        // ---- Ricezione UDP ----
        int n = receiver.receive(buf, sizeof(buf));
        if (n <= 0) continue;
        if (n < 700) continue;
        if (buf[0] != 1) continue;

        const unsigned char* pdus = buf + 3;

        LWI01 lwi;
        lwi.decode(pdus + 677);

        Lichthinten01 lh;
        lh.decode(pdus + 605);

        ChassisState& slot = mem.host_ring[writeIndex];
        slot.steering_angle = lwi.angle;
        slot.steering_speed = lwi.speed;
        slot.lights_rear    = lh.compute_mask();

        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        slot.timestamp_ns =
            uint64_t(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;

        int processedIndex = writeIndex;
        writeIndex = (writeIndex + 1) % mem.capacity;

        // ---- Lancio GPU kernel ----
        launch_kernel(mem.device_ring, processedIndex);

        // ---- Invio CSV via UDP (angle,speed) ----
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3)
           << slot.steering_angle << "," << slot.steering_speed;
        std::string msg = ss.str();
        sender.send(msg);

        // ---- Fine loop: calcolo tempo in millisecondi ----
        auto loop_end = Clock::now();
        double loop_ms = std::chrono::duration<double, std::milli>(loop_end - loop_start).count();

        std::cout << "[CPU] Index=" << processedIndex
                  << " | " << msg
                  << " | Loop time: " << std::fixed << std::setprecision(3) << loop_ms << " ms\n";
    }

    return 0;
}
