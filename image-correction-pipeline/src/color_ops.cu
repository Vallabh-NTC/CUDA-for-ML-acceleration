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

// Fast 3x3 Laplacian on Y (for unsharp)
__device__ __forceinline__ float laplacian3x3(const uint8_t* Y, int W, int H, int pitch, int x, int y)
{
    int xm1 = x>0 ? x-1 : 0, xp1 = x+1<W?x+1:W-1;
    int ym1 = y>0 ? y-1 : 0, yp1 = y+1<H?y+1:H-1;

    float c  = (float)Y[y*pitch + x];
    float n  = (float)Y[ym1*pitch + x];
    float s  = (float)Y[yp1*pitch + x];
    float w  = (float)Y[y*pitch + xm1];
    float e  = (float)Y[y*pitch + xp1];

    return (n + s + w + e - 4.f * c);
}

// Nonlinear contrast S-curve around mid.
// c in [-1..+1]. For c -> +1 it approaches a hard threshold @0.5 (asymptotically).
// For c -> -1 it compresses toward mid-gray.
__device__ __forceinline__ float scontrast_curve(float t, float c)
{
    t = clamp(t, 0.f, 1.f);
    if (c >= 0.f) {
        // Boost contrast: exponent g in (0,1].
        float g = fmaxf(1e-3f, 1.f - 0.999f * c); // c=1 -> g≈0 (very steep but continuous)
        if (t < 0.5f) return 0.5f * powf(2.f * t, g);
        else          return 1.f - 0.5f * powf(2.f * (1.f - t), g);
    } else {
        // Reduce contrast: exponent g in [1, 5].
        float g = 1.f + 4.f * (-c);               // c=-1 -> g=5 (flat)
        if (t < 0.5f) return 0.5f * powf(2.f * t, g);
        else          return 1.f - 0.5f * powf(2.f * (1.f - t), g);
    }
}

// ============================================================================
// Tone + color kernel (NV12 in-place)
//   - Brightness full-range at the end (±1 => white/black).
//   - Stronger shadows/highlights response.
//   - Whites sign: >0 raises headroom, <0 lowers ceiling.
//   - Contrast: UI 0.50..1.80 → [-1..+1]; **no hard threshold at max**.
//   - Saturation: neutral at 1.0; >1 boosts aggressively (up to 4x).
// ============================================================================
__global__ void toneSat_kernel(
    uint8_t* __restrict__ Y, int W, int H, int pitchY,
    uint8_t* __restrict__ UV,            int pitchUV,
    float gain, float contrast_factor, float highlights, float shadows, float whites,
    float gamma, float sat_factor, int tv_range,
    float brightness, float brilliance, float sharp_amt)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int Ymin = tv_range ? 16 : 0;
    const int Ymax_base = tv_range ? 235 : 255;

    // Whites: >0 raises headroom, <0 lowers ceiling.
    int Ymax = clamp((int)(Ymax_base + whites * 30.f), Ymin + 16, 255);

    // --- Luma ---
    float yf = (float)Y[y * pitchY + x];
    yf *= gain;

    // Normalize to [0..1]
    float t = (yf - (float)Ymin) / (float)(Ymax - Ymin);
    t = clamp(t, 0.0f, 1.0f);

    // Shadows: stronger effect on blacks
    float gamma_sh = clamp(1.0f - 0.8f * shadows, 0.2f, 2.0f);
    float w_sh     = 1.0f - smoothstepf(0.35f, 0.75f, t);
    float t_sh     = powf(t, gamma_sh);
    t = t * (1.0f - w_sh) + t_sh * w_sh;

    // Highlights: earlier knee if highlights<0
    float knee = clamp(0.65f + 0.30f * highlights, 0.20f, 0.95f);
    float soft = clamp(0.12f + 0.28f * (-highlights), 0.05f, 0.50f);
    if (t > knee) {
        float u = (t - knee) / soft;
        float c = 1.0f - __expf(-u);
        t = knee + c * (1.0f - knee);
    }

    // Brilliance: mid-boost, tapered near tails
    float w_mid = smoothstepf(0.12f, 0.50f, t) * (1.0f - smoothstepf(0.50f, 0.88f, t));
    t = clamp(t + 0.50f * brilliance * (w_mid - 0.33f), 0.0f, 1.0f);

    // -------- Nonlinear Contrast (map 0.50..1.80 → [-1..+1]) --------
    float c01;
    if (contrast_factor >= 1.f) {
        c01 = (contrast_factor - 1.f) / (1.80f - 1.0f);       // [0..1]
    } else {
        c01 = - ((1.0f - contrast_factor) / (1.0f - 0.50f));  // [-1..0]
    }
    c01 = clamp(c01, -1.f, 1.f);

    c01 = fminf(c01, 0.97f);
    t = scontrast_curve(t, c01);    // always smooth S-curve

    // Global gamma
    t = powf(t, 1.0f / gamma);


    t = clamp(t + brightness, 0.0f, 1.0f);

    // Back to luma (pre-sharpen)
    float y_pre = (float)Ymin + t * (float)(Ymax - Ymin);
    y_pre = clamp(y_pre, (float)Ymin, (float)Ymax);

    // Unsharp on Y (moderate)
    if (sharp_amt > 1e-6f) {
        float lap = laplacian3x3(Y, W, H, pitchY, x, y);
        float y_sharp = y_pre + sharp_amt * lap * 0.50f;
        y_pre = clamp(y_sharp, (float)Ymin, (float)Ymax);
    }

    Y[y * pitchY + x] = (uint8_t)(y_pre + 0.5f);

    // --- Chroma (once per 2x2) ---
    if ((x % 2 == 0) && (y % 2 == 0)) {
        const int idx = (y / 2) * pitchUV + (x & ~1);
        float U = UV[idx + 0], V = UV[idx + 1];

        // Saturation: neutral at 1.0. >1 boosts aggressively (up to 4x).
        float sat_eff = (sat_factor >= 1.f)
                        ? fminf(4.f, 1.f + 3.0f * (sat_factor - 1.f)) // 1.0→1.0, 1.5→2.5, 2.0→4.0 (cap)
                        : sat_factor;                                  // <1 desaturates

        // Slight protection near whites (keep colors from blowing out)
        float t_hi = (y_pre - (float)Ymin) / (float)(Ymax - Ymin);
        float hi = clamp((t_hi - 0.85f) / 0.15f, 0.0f, 1.0f);
        float s  = sat_eff * (1.0f - 0.2f * hi);

        U = 128.f + s * (U - 128.f);
        V = 128.f + s * (V - 128.f);

        UV[idx + 0] = (uint8_t)clamp(U, 0.f, 255.f);
        UV[idx + 1] = (uint8_t)clamp(V, 0.f, 255.f);
    }
}

// ============================================================================
// Host launcher: keep ranges aligned to your current JSON/UI
//   contrast:   0.50 .. 1.80  (UI 0..100; at 100 we cap to 0.97 internally)
//   saturation: 0.00 .. 4.00  (neutral 1.0)
//   brightness: -1.00.. +1.00 (full white/black)
// ============================================================================
void launch_tone_saturation_nv12(
    uint8_t* dY, int W, int H, int pitchY,
    uint8_t* dUV,            int pitchUV,
    const ColorParams& pin,
    cudaStream_t stream)
{
    if (!pin.enable) return;

    ColorParams p = pin;
    p.exposure_ev = clamp(p.exposure_ev, -2.0f,  2.0f);
    p.contrast    = clamp(p.contrast,     0.50f, 1.80f); // UI maps to this range; kernel caps smoothly
    p.highlights  = clamp(p.highlights,  -1.0f,  1.0f);
    p.shadows     = clamp(p.shadows,     -1.0f,  1.0f);
    p.whites      = clamp(p.whites,      -1.0f,  1.0f);
    p.gamma       = clamp(p.gamma,        0.50f, 2.00f);
    p.saturation  = clamp(p.saturation,   0.00f, 4.00f); // neutral at 1.0
    p.brightness  = clamp(p.brightness,  -1.0f,  1.0f);  // full-range final offset
    p.brilliance  = clamp(p.brilliance,  -1.0f,  1.0f);
    p.sharpness   = clamp(p.sharpness,    0.0f,  1.0f);

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
