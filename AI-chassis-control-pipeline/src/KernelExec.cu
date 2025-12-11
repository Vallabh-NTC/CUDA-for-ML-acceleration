#include "ChassisState.hpp"
#include <cuda_runtime.h>

// ---------------------------------------------------------
// GPU Kernel
// ---------------------------------------------------------
__global__ void kernel_process(ChassisState* ring, int index)
{
    // placeholder logic – dimostra che leggiamo/scriviamo memoria
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // NIENTE printf qui: è troppo lenta a 200 Hz
        ring[index].steering_angle *= 1.01f; // esempio di write-back
    }
}

// ---------------------------------------------------------
// CPU-callable wrapper
// ---------------------------------------------------------
extern "C" void launch_kernel(ChassisState* ring, int index)
{
    kernel_process<<<1,1>>>(ring, index);
    cudaDeviceSynchronize();   // se vuoi ancora più velocità puoi toglierla
}
