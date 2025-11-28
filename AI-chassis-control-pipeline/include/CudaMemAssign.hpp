#pragma once
#include "ChassisState.hpp"
#include <cuda_runtime.h>

class CudaMemAssign {
public:
    int capacity;
    ChassisState* host_ring;
    ChassisState* device_ring;

    CudaMemAssign(int cap) : capacity(cap)
    {
        size_t bytes = sizeof(ChassisState) * capacity;

        // Pinned mapped memory (zero copy)
        cudaHostAlloc((void**)&host_ring, bytes, cudaHostAllocMapped);
        cudaHostGetDevicePointer((void**)&device_ring, host_ring, 0);
    }

    ~CudaMemAssign() {
        cudaFreeHost(host_ring);
    }
};
