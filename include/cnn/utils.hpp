#pragma once
#include <cstddef>

namespace cnn {

// Minimal conv forward stub using cuDNN
void conv_forward(const float* input,
                  const float* filter,
                  float* output,
                  int N, int C, int H, int W,
                  int K, int R, int S,
                  int stride_h, int stride_w,
                  int pad_h, int pad_w);

// Minimal GEMM forward using cuBLASLt
void gemm_forward(const float* A, const float* B, float* C,
                  int M, int N, int K);

} // namespace cnn
