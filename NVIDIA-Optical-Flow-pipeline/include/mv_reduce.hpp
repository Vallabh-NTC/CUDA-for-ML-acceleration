// include/mv_reduce.hpp
#pragma once
#include <cstdint>

// Output of the pure reduction (no gating here).
// mean_* are in px/frame.
struct MVPureOut
{
    int   count;

    float mean_dx;   // px/frame
    float mean_dy;   // px/frame
    float res_mag;   // px/frame (hypot(mean_dx, mean_dy))

    // Quality metrics (dispersion in ROI)
    float std_dx;    // px/frame
    float std_dy;    // px/frame
    float std_mag;   // px/frame (hypot(std_dx, std_dy))

    float speed_mps; // m/s computed from res_mag, pxPerMeter and dtSec
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

// Computes mean dx/dy, std dx/dy (population std), resultant magnitude and speed.
void mv_reduce_pure_cuda(const int16_t *mvPtr,
                         const MVPureParams *params,
                         MVPureOut *d_out);

#ifdef __cplusplus
}
#endif
