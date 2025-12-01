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

// GPU launcher function
extern "C" void launch_kernel(ChassisState* ring, int index);

int main()
{
    CudaMemAssign mem(64);
    int writeIndex = 0;

    UdpReceiver receiver(1500);

    // -------------------------
    // NEW: UDP sender to Windows
    // -------------------------
    UdpSender sender("192.168.1.100", 1600);   // <-- CHANGE THIS TO YOUR WINDOWS IP

    unsigned char buf[4096];

    std::cout << "Start decoder + GPU pipeline\n";

    while (true)
    {
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

        launch_kernel(mem.device_ring, processedIndex);

        // ------------------ Debug line ------------------
        std::ostringstream ss;
        ss << "[CPU] Index=" << processedIndex
           << " | Angle=" << slot.steering_angle
           << " | Speed=" << slot.steering_speed
           << " | Mask=0x" << std::hex << slot.lights_rear << std::dec;

        std::string debug = ss.str();

        std::cout << debug << "\n";        // local print
        sender.send(debug);                // send to Windows
    }

    return 0;
}
