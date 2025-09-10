#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "color_ops.cuh"

namespace icp {

template<typename T>
__host__ __device__ static inline T clamp(T v, T lo, T hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Smoothstep helper for weights
__device__ __forceinline__ float smoothstepf(float e0, float e1, float x)
{
    float t = clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.f - 2.f * t);
}

// Tone map + saturation (manual-only)
__global__ void toneSat_kernel(
    uint8_t* __restrict__ Y, int W, int H, int pitchY,
    uint8_t* __restrict__ UV,            int pitchUV,
    float gain, float contrast, float highlights, float shadows, float whites,
    float gamma, float sat, int tv_range)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int Ymin = tv_range ? 16 : 0;
    const int Ymax_base = tv_range ? 235 : 255;

    // White ceiling adjustment: lower if whites<0, slightly raise if >0
    int Ymax = clamp(Ymax_base - (int)(whites * 30.f), Ymin + 32, 255);

    // --- Luma ---
    float yf = (float)Y[y * pitchY + x];

    // Manual exposure gain (already clamped host-side)
    yf *= gain;

    // Normalize to [0..1]
    float t = (yf - (float)Ymin) / (float)(Ymax - Ymin);
    t = clamp(t, 0.0f, 1.0f);

    // Shadows: gamma near blacks only
    float gamma_sh = 1.0f - 0.5f * shadows;  // in [0.5 .. 1.5]
    float w_sh     = 1.0f - smoothstepf(0.35f, 0.75f, t);
    float t_sh     = powf(t, gamma_sh);
    t = t * (1.0f - w_sh) + t_sh * w_sh;

    // Highlights: soft knee compression (earlier knee if highlights<0)
    float knee = clamp(0.75f + 0.25f * highlights, 0.45f, 0.95f);
    float soft = clamp(0.20f + 0.20f * (-highlights), 0.05f, 0.40f);
    if (t > knee) {
        float u = (t - knee) / soft;   // distance into knee
        float c = 1.0f - __expf(-u);   // smooth roll-off
        t = knee + c * (1.0f - knee);
    }

    // Contrast (S-curve around mid)
    t = clamp(0.5f + (t - 0.5f) * contrast, 0.0f, 1.0f);

    // Optional global gamma
    t = powf(t, 1.0f / gamma);

    // Back to luma
    yf = (float)Ymin + t * (float)(Ymax - Ymin);
    yf = clamp(yf, (float)Ymin, (float)Ymax);
    Y[y * pitchY + x] = (uint8_t)(yf + 0.5f);

    // --- Chroma (once per 2x2) ---
    if ((x % 2 == 0) && (y % 2 == 0)) {
        const int idx = (y / 2) * pitchUV + (x & ~1);
        float U = UV[idx + 0], V = UV[idx + 1];

        // Reduce saturation near whites (avoid pink skies)
        float hi = clamp((t - 0.80f) / 0.20f, 0.0f, 1.0f);
        float s  = sat * (1.0f - 0.6f * hi);

        U = 128.f + s * (U - 128.f);
        V = 128.f + s * (V - 128.f);

        UV[idx + 0] = (uint8_t)clamp(U, 0.f, 255.f);
        UV[idx + 1] = (uint8_t)clamp(V, 0.f, 255.f);
    }
}

void launch_tone_saturation_nv12(
    uint8_t* dY, int W, int H, int pitchY,
    uint8_t* dUV,            int pitchUV,
    const ColorParams& pin,
    cudaStream_t stream)
{
    if (!pin.enable) return;

    // Clamp params to safe ranges
    ColorParams p = pin;
    p.exposure_ev = clamp(p.exposure_ev, -2.0f, 2.0f);
    p.contrast    = clamp(p.contrast,    0.50f, 1.80f);
    p.highlights  = clamp(p.highlights, -1.0f, 1.0f);
    p.shadows     = clamp(p.shadows,    -1.0f, 1.0f);
    p.whites      = clamp(p.whites,     -1.0f, 1.0f);
    p.gamma       = clamp(p.gamma,       0.70f, 1.30f);
    p.saturation  = clamp(p.saturation,  0.50f, 1.50f);

    // Manual exposure → linear gain
    const float gain = powf(2.0f, p.exposure_ev);

    dim3 b(16, 16), g((W + b.x - 1) / b.x, (H + b.y - 1) / b.y);
    toneSat_kernel<<<g, b, 0, stream>>>(
        dY, W, H, pitchY,
        dUV,      pitchUV,
        gain, p.contrast, p.highlights, p.shadows, p.whites,
        p.gamma, p.saturation, p.tv_range ? 1 : 0);
}

} // namespace icp
