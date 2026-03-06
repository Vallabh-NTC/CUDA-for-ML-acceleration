#pragma once
// nv12_to_rgb_fp32.hpp (rinominato concettualmente — usa float32)
// CUDA kernel: NV12 → RGB float32 tensor [1, 3, H, W] normalized [0,1]

#include <cuda_runtime.h>
#include <cstdint>

void nv12_to_rgb_fp16(
    const uint8_t *d_y,
    const uint8_t *d_uv,
    int            pitchY,
    int            pitchUV,
    int            W,
    int            H,
    float         *d_out,       // float32, not fp16
    cudaStream_t   stream = 0);