#include "ChassisState.hpp"
#include <cuda_runtime.h>
#include <cstdio>

// ---------------------------------------------------------
// GPU Kernel
// ---------------------------------------------------------
__global__ void kernel_process(ChassisState* ring, int index)
{
    // placeholder logic – demonstrate we can read/write memory
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("[GPU] Processing index %d | angle=%f | speed=%f\n",
               index,
               ring[index].steering_angle,
               ring[index].steering_speed);

        // Example write-back
        ring[index].steering_angle *= 1.01f; // +1%
    }
}

// ---------------------------------------------------------
// CPU-callable wrapper
// ---------------------------------------------------------
extern "C" void launch_kernel(ChassisState* ring, int index)
{
    kernel_process<<<1,1>>>(ring, index);
    cudaDeviceSynchronize();
}
