#pragma once
// raft_infer.hpp — TensorRT wrapper for RAFT Small (float32 I/O)

#include <cuda_runtime.h>
#include <string>

class RaftInfer
{
public:
    RaftInfer()  = default;
    ~RaftInfer() { destroy(); }

    bool init(const std::string &engine_path, int W, int H);

    // d_frame_prev/curr : float32 [1,3,H,W] on GPU
    // d_flow_out        : float32 [1,2,H,W] on GPU
    bool infer(const float *d_frame_prev,
               const float *d_frame_curr,
               float       *d_flow_out,
               cudaStream_t stream = 0);

    void destroy();

    bool isReady() const { return m_ready; }
    int  W()       const { return m_W; }
    int  H()       const { return m_H; }

private:
    bool  m_ready   = false;
    int   m_W       = 0;
    int   m_H       = 0;
    void *m_runtime = nullptr;
    void *m_engine  = nullptr;
    void *m_context = nullptr;
};