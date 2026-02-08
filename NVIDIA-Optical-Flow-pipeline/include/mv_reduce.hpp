#pragma once
#include <cstdint>

// Persistent per-instance device state stored on GPU.
struct DevEmaState
{
    float speedEma;   // EMA filtered speed (m/s)
    int   haveEma;    // 0/1
    float lastOut;    // last output (hold on bad frames)

    // Last global motion direction (px/frame) for overlay resultant arrow.
    float lastResDx;
    float lastResDy;
};

// Per-frame reduction outputs (computed on GPU).
struct MVReduceOut
{
    int   count;       // robust samples count (after mag band-pass)
    float sum_dx;
    float sum_dy;
    float sum_mag;
    float sum_mag2;
    float max_mag;     // max magnitude across sampled points
};

// Parameters passed to GPU reducer.
struct MVParams
{
    int mvW, mvH;
    int mvPitchBytes;  // pitch in bytes (CUDA pitch-linear)
    int grid;          // MV cell -> pixels

    int x0, x1, y0, y1; // ROI in MV cells
    int step;           // sampling step in MV cells

    float minMag;
    float maxMag;

    int   minSamples;
    float cohMin;
    float stdMax;
    float tailRatio;
    float tailMinAbs;

    float pxPerMeter;
    float dtSec;

    float alphaHi;
    float alphaLo;

    float cohExcellent;
    float stdExcellentMax;
};

#ifdef __cplusplus
extern "C" {
#endif

void mv_reduce_gating_ema_cuda(const int16_t *mvPtr,
                              DevEmaState   *devState,
                              const MVParams *params,
                              float         *d_outSpeed,
                              MVReduceOut   *d_outDiag);

#ifdef __cplusplus
}
#endif
