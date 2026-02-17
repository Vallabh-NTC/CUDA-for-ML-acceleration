// include/mv_spike_filter.hpp
#pragma once

#include <stdint.h>
#include "mv_reduce.hpp"   // uses MVPureOut

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MVSpikeFilterParams
{
    float spike_kmh;      // jump considered a spike (km/h)
    float ok_kmh;         // kept for compatibility; currently not used
    int   stable_frames;  // how many frames the jump must persist to be accepted
} MVSpikeFilterParams;

typedef struct MVSpikeFilterState
{
    int   have;            // do we already have a last-good?
    int   stable_cnt;      // persistence counter
    float last_speed_kmh;  // last accepted speed (km/h)

    // last accepted vector
    float last_dx;
    float last_dy;
    float last_mag;
    float last_speed_mps;
} MVSpikeFilterState;

void mv_spike_filter_cuda(const MVPureOut *d_in,
                          const MVSpikeFilterParams *h_params,
                          MVSpikeFilterState *d_state,
                          MVPureOut *d_out);

#ifdef __cplusplus
}
#endif
