#include "UdpReceiver.hpp"
#include "LWI01.hpp"
#include "Lichthinten01.hpp"
#include "ChassisState.hpp"
#include "CudaMemAssign.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>

// GPU launcher function
extern "C" void launch_kernel(ChassisState* ring, int index);

int main()
{
    // ---- Allocate unified/pinned CUDA memory ----
    CudaMemAssign mem(64);   // 64-frame ring buffer
    int writeIndex = 0;

    UdpReceiver receiver(1500);
    unsigned char buf[4096];

    std::cout << "Start decoder + GPU pipeline\n";

    while (true)
    {
        int n = receiver.receive(buf, sizeof(buf));
        if (n <= 0) continue;
        if (n < 700) continue;

        if (buf[0] != 1) continue;

        const unsigned char* pdus = buf + 3;

        // ------------------ Decode signals ------------------
        LWI01 lwi;
        lwi.decode(pdus + 677);

        Lichthinten01 lh;
        lh.decode(pdus + 605);

        // ---------------- Write into CUDA-shared ring buffer ----------------
        ChassisState& slot = mem.host_ring[writeIndex];

        slot.steering_angle  = lwi.angle;   // fixed names
        slot.steering_speed  = lwi.speed;

        // Compute a simple bitmask for rear lights
        slot.lights_rear = lh.compute_mask();

        // Timestamp
        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        slot.timestamp_ns =
            uint64_t(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;

        // rotate ring index
        int processedIndex = writeIndex;
        writeIndex = (writeIndex + 1) % mem.capacity;

        // ------------------ Launch GPU kernel -------------------
        launch_kernel(mem.device_ring, processedIndex);

        // ------------------ Debug prints ------------------------
        std::cout << "[CPU] Index=" << processedIndex
                  << " | Angle=" << slot.steering_angle
                  << " | Speed=" << slot.steering_speed
                  << " | Mask=0x" << std::hex << slot.lights_rear << std::dec
                  << "\n";
    }

    return 0;
}
