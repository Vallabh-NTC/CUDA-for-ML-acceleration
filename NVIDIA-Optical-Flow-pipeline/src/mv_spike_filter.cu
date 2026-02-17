// src/mv_spike_filter.cu
#include "mv_spike_filter.hpp"
#include <cuda_runtime.h>
#include <math.h>

__global__ void mv_spike_filter_kernel(const MVPureOut *in,
                                       MVSpikeFilterParams params,
                                       MVSpikeFilterState *st,
                                       MVPureOut *out)
{
    // Single-thread stateful filter
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    MVPureOut x = *in;

    // speed -> km/h
    float cur_kmh = x.speed_mps * 3.6f;
    if (!isfinite(cur_kmh)) cur_kmh = 0.0f;

    // First sample: accept
    if (!st->have) {
        st->have = 1;
        st->stable_cnt = 0;
        st->last_speed_kmh = cur_kmh;

        st->last_dx = x.mean_dx;
        st->last_dy = x.mean_dy;
        st->last_mag = x.res_mag;
        st->last_speed_mps = x.speed_mps;

        *out = x;
        return;
    }

    float last_kmh = st->last_speed_kmh;
    float diff = fabsf(cur_kmh - last_kmh);

    // If big jump => possible spike
    if (params.spike_kmh > 0.0f && diff >= params.spike_kmh) {

        // Count persistence
        st->stable_cnt += 1;

        // Until stable_frames reached: HOLD last accepted values
        if (st->stable_cnt < params.stable_frames) {
            MVPureOut y = x;
            y.mean_dx   = st->last_dx;
            y.mean_dy   = st->last_dy;
            y.res_mag   = st->last_mag;
            y.speed_mps = st->last_speed_mps;
            *out = y;
            return;
        }

        // Persisted enough => accept new regime
        st->stable_cnt = 0;

    } else {
        // Not a big jump => reset counter
        st->stable_cnt = 0;
    }

    // Accept update
    st->last_speed_kmh = cur_kmh;
    st->last_dx = x.mean_dx;
    st->last_dy = x.mean_dy;
    st->last_mag = x.res_mag;
    st->last_speed_mps = x.speed_mps;

    *out = x;
}

void mv_spike_filter_cuda(const MVPureOut *d_in,
                          const MVSpikeFilterParams *h_params,
                          MVSpikeFilterState *d_state,
                          MVPureOut *d_out)
{
    MVSpikeFilterParams p = *h_params;
    mv_spike_filter_kernel<<<1,1>>>(d_in, p, d_state, d_out);
}
