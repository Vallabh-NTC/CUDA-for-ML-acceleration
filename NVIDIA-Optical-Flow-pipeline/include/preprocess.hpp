#pragma once
// preprocess.hpp
// GPU sharpening (unsharp mask) on float32 RGB tensor [1,3,H,W]
// Applied before RAFT inference to recover texture lost to motion blur.
//
// Unsharp mask formula:
//   sharpened = original + strength * (original - gaussian_blur)
//
// strength env var: RAFT_SHARP (default 1.5)

#include <cuda_runtime.h>

// Sharpen float32 RGB tensor in place.
// d_frame : float32 [1,3,H,W] — modified in place
// W, H    : frame dimensions
// strength: sharpening amount (0=none, 1=moderate, 2=strong)
// stream  : CUDA stream
void preprocess_sharpen(
    float        *d_frame,
    int           W,
    int           H,
    float         strength,
    cudaStream_t  stream = 0);