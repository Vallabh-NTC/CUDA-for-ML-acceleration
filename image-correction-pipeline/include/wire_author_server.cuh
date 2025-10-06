// wire_author_server.cuh
#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

// Start/stop the network+device module (safe to call multiple times; refcounted).
// If ctx==nullptr, the module will retain the CUDA primary context itself.
void wire_author_start(CUcontext ctx, uint16_t tcp_port);
void wire_author_stop();

// Called each frame (after your rectify/crop step). If the server thread has
// requested a snapshot, this captures NV12→RGBA into a pinned host buffer and
// signals the thread to send it.
void wire_author_snapshot_nv12_if_needed(
    const uint8_t* dY, int pitchY,
    const uint8_t* dUV, int pitchUV,
    int W, int H, cudaStream_t stream);

// Optional: query the currently active device-resident mask (for debug).
// Returns true if a mask is available; outputs are filled.
bool wire_author_get_active_mask(
    CUdeviceptr* dMask, int* pitchBytes,
    float* dx, float* dy,
    uint32_t* version /*optional, can be nullptr*/);
