#pragma once
// flow_reduce.hpp
// GPU kernel: compute median u,v over ROI from RAFT flow field.
// Output is a small struct written directly on GPU — no CPU transfer.

#include <cuda_runtime.h>

struct FlowResult {
    float mean_u;   // mean horizontal displacement (px/frame)
    float mean_v;   // mean vertical   displacement (px/frame)
};

// Compute mean u,v over ROI pixels (sampled every `step` pixels).
// d_flow   : float32 [1,2,H,W] RAFT output
// d_result : GPU pointer to FlowResult — written in place
void flow_reduce(
    const float  *d_flow,
    int           W,
    int           H,
    int           roi_x0,
    int           roi_x1,
    int           roi_y0,
    int           roi_y1,
    int           step,
    FlowResult   *d_result,
    cudaStream_t  stream = 0);