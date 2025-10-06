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

// 3x3 unsharp (fast, separable-ish): simple Laplacian on Y then add amount
__device__ __forceinline__ float laplacian3x3(const uint8_t* Y, int W, int H, int pitch, int x, int y)
{
    // Clamp sampling at borders
    int xm1 = x>0 ? x-1 : 0, xp1 = x+1<W?x+1:W-1;
    int ym1 = y>0 ? y-1 : 0, yp1 = y+1<H?y+1:H-1;

    float c  = (float)Y[y*pitch + x];
    float n  = (float)Y[ym1*pitch + x];
    float s  = (float)Y[yp1*pitch + x];
    float w  = (float)Y[y*pitch + xm1];
    float e  = (float)Y[y*pitch + xp1];

    // 4-neighborhood Laplacian
    return (n + s + w + e - 4.f * c);
}

// Tone map + saturation + brightness/brilliance + optional unsharp
__global__ void toneSat_kernel(
    uint8_t* __restrict__ Y, int W, int H, int pitchY,
    uint8_t* __restrict__ UV,            int pitchUV,
    float gain, float contrast, float highlights, float shadows, float whites,
    float gamma, float sat, int tv_range,
    float brightness, float brilliance, float sharp_amt)
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

    // Brightness: small linear mid-shift
    t = clamp(t + 0.15f * brightness, 0.0f, 1.0f);

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

    // Brilliance: mid-boost with roll-off near 0/1
    // weight peaks at mid-gray, fades to 0 near extremes
    float w_mid = smoothstepf(0.15f, 0.5f, t) * (1.0f - smoothstepf(0.5f, 0.85f, t));
    t = clamp(t + 0.25f * brilliance * (w_mid - 0.33f), 0.0f, 1.0f);

    // Contrast (S-curve around mid)
    t = clamp(0.5f + (t - 0.5f) * contrast, 0.0f, 1.0f);

    // Optional global gamma
    t = powf(t, 1.0f / gamma);

    // Back to luma (pre-sharpen)
    float y_pre = (float)Ymin + t * (float)(Ymax - Ymin);
    y_pre = clamp(y_pre, (float)Ymin, (float)Ymax);

    // Optional unsharp on Y (small, safe range)
    if (sharp_amt > 1e-6f) {
        float lap = laplacian3x3(Y, W, H, pitchY, x, y);
        // scale Laplacian to ~[-255..255], normalize and apply amount
        float y_sharp = y_pre + sharp_amt * lap * 0.25f;
        y_pre = clamp(y_sharp, (float)Ymin, (float)Ymax);
    }

    Y[y * pitchY + x] = (uint8_t)(y_pre + 0.5f);

    // --- Chroma (once per 2x2) ---
    if ((x % 2 == 0) && (y % 2 == 0)) {
        const int idx = (y / 2) * pitchUV + (x & ~1);
        float U = UV[idx + 0], V = UV[idx + 1];

        // Reduce saturation near whites (avoid pink skies)
        float t_hi = (y_pre - (float)Ymin) / (float)(Ymax - Ymin);
        float hi = clamp((t_hi - 0.80f) / 0.20f, 0.0f, 1.0f);
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
    p.brightness  = clamp(p.brightness, -1.0f, 1.0f);
    p.brilliance  = clamp(p.brilliance, -1.0f, 1.0f);
    p.sharpness   = clamp(p.sharpness,   0.0f, 1.0f);

    // Manual exposure → linear gain
    const float gain = powf(2.0f, p.exposure_ev);

    dim3 b(16, 16), g((W + b.x - 1) / b.x, (H + b.y - 1) / b.y);
    toneSat_kernel<<<g, b, 0, stream>>>(
        dY, W, H, pitchY,
        dUV,      pitchUV,
        gain, p.contrast, p.highlights, p.shadows, p.whites,
        p.gamma, p.saturation, p.tv_range ? 1 : 0,
        p.brightness, p.brilliance, p.sharpness);
}

} // namespace icp
