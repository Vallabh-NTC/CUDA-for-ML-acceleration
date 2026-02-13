// include/mv_spike_filter.hpp
#pragma once

#include <stdint.h>
#include "mv_reduce.hpp"   // <-- usa MVPureOut definito qui

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MVSpikeFilterParams
{
    float spike_kmh;      // salto considerato spike (km/h)
    float ok_kmh;         // (opzionale) tolleranza, qui non usata per gating complesso
    int   stable_frames;  // quanti frame deve persistere per accettare il nuovo regime
} MVSpikeFilterParams;

typedef struct MVSpikeFilterState
{
    int   have;            // abbiamo già un last-good?
    int   stable_cnt;      // contatore persistenza spike
    float last_speed_kmh;  // ultimo speed accettato (km/h)

    // ultimo vettore accettato
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
