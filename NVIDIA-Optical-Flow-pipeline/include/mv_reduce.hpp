#pragma once
#include <cstdint>

// Output of the pure reduction (no gating, no EMA).
struct MVPureOut
{
    int   count;
    float mean_dx;   // px/frame
    float mean_dy;   // px/frame
    float res_mag;   // px/frame (hypot(mean_dx, mean_dy))
    float speed_mps; // m/s
};

// Parameters for pure reduction.
struct MVPureParams
{
    int mvW, mvH;
    int mvPitchBytes; // pitch in bytes (CUDA pitch-linear)
    int grid;         // MV cell -> pixels (OFA grid)

    // ROI in MV cells [x0,x1), [y0,y1)
    int x0, x1, y0, y1;

    // Sampling step in MV cells (1 = all, 2 = every 2 cells, ...)
    int step;

    // Conversion to speed
    float pxPerMeter;
    float dtSec;
};

#ifdef __cplusplus
extern "C" {
#endif

// Pure reduction: computes mean dx/dy over ROI and speed from resultant.
// mvPtr is pitch-linear int16 interleaved dx,dy in S10.5 (same as your OFA output after VIC BL->PL).
void mv_reduce_pure_cuda(const int16_t *mvPtr,
                         const MVPureParams *params,
                         MVPureOut *d_out);

#ifdef __cplusplus
}
#endif
