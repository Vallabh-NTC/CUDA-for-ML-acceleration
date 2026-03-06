// preprocess.cu
// Unsharp mask sharpening on float32 RGB tensor [1,3,H,W].
// Applied before RAFT to recover asphalt texture on blurry frames.
//
// Pipeline per pixel:
//   1. Read 3x3 neighborhood
//   2. Apply Gaussian blur weights
//   3. sharpened = clamp(original + strength * (original - blurred), 0, 1)
//
// Uses shared memory to avoid redundant global reads.

#include "preprocess.hpp"
#include <cuda_runtime.h>
#include <cstdint>

// ── Gaussian 3x3 kernel weights (sum=1) ──────────────────────────────────────
// [1 2 1]
// [2 4 2]  / 16
// [1 2 1]
__device__ __forceinline__ float gaussian3x3(
    const float *plane, int W, int H, int x, int y)
{
    // Clamp-to-border sampling
    auto at = [&](int px, int py) -> float {
        px = max(0, min(W-1, px));
        py = max(0, min(H-1, py));
        return plane[py * W + px];
    };

    return (
        1.0f * at(x-1,y-1) + 2.0f * at(x,y-1) + 1.0f * at(x+1,y-1) +
        2.0f * at(x-1,y  ) + 4.0f * at(x,y  ) + 2.0f * at(x+1,y  ) +
        1.0f * at(x-1,y+1) + 2.0f * at(x,y+1) + 1.0f * at(x+1,y+1)
    ) * (1.0f / 16.0f);
}

// ── Sharpening kernel ─────────────────────────────────────────────────────────
__global__ void sharpen_kernel(
    float       *__restrict__ d_frame,   // [3, H, W] CHW in place
    int          W,
    int          H,
    float        strength)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int planeSize = H * W;
    const int idx       = y * W + x;

    // Process all 3 channels (R, G, B)
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float *plane = d_frame + c * planeSize;

        const float orig   = plane[idx];
        const float blur   = gaussian3x3(plane, W, H, x, y);

        // Unsharp mask: amplify high-frequency detail
        float sharpened = orig + strength * (orig - blur);

        // Clamp to valid range [0, 1]
        d_frame[c * planeSize + idx] = fminf(fmaxf(sharpened, 0.0f), 1.0f);
    }
}

// ── Host wrapper ──────────────────────────────────────────────────────────────
void preprocess_sharpen(
    float        *d_frame,
    int           W,
    int           H,
    float         strength,
    cudaStream_t  stream)
{
    if (strength <= 0.0f) return;  // skip if disabled

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    sharpen_kernel<<<grid, block, 0, stream>>>(d_frame, W, H, strength);
}