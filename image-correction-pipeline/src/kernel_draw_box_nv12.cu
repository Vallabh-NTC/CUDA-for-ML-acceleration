#include <cstdio>
#include <cuda_runtime.h>
#include "kernel_draw_box_nv12.cuh"

namespace draw {

// ------------------------------------------------------------------
// Kernel: disegna il bordo di un rettangolo su NV12
//  - dY  : piano Y (luma), passo = pitch
//  - dUV : piano UV interleaved, passo = pitch, size = H/2 righe
//  - W,H : dimensione frame
//  - x0,y0,w,h : ROI
// ------------------------------------------------------------------
__global__ void draw_box_nv12_kernel(
    uint8_t* dY, uint8_t* dUV,
    int W, int H, int pitch,
    int x0, int y0, int w, int h,
    uint8_t Yval, uint8_t Uval, uint8_t Vval)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    // spessore del bordo
    const int t = 4;

    bool inside_x = (x >= x0) && (x < x0 + w);
    bool inside_y = (y >= y0) && (y < y0 + h);

    bool on_border =
        inside_x && inside_y &&
        (
            x < x0 + t ||                    // bordo sinistro
            x >= (x0 + w - t) ||             // bordo destro
            y < y0 + t ||                    // bordo alto
            y >= (y0 + h - t)                // bordo basso
        );

    if (!on_border) return;

    // ---------------- Y plane ----------------
    dY[y * pitch + x] = Yval;

    // ---------------- UV plane ----------------
    // In NV12 una coppia (U,V) descrive un blocco 2x2 di pixel
    // Scriviamo solo per pixel (x,y) con x,y pari per evitare race.
    if ((x & 1) == 0 && (y & 1) == 0)
    {
        int uv_row = y / 2;
        int uv_col = x;          // stesso pitch della Y

        // sicurezza: limiti UV
        if (uv_row < (H / 2) && (uv_col + 1) < pitch)
        {
            int uv_idx = uv_row * pitch + uv_col;
            dUV[uv_idx + 0] = Uval;
            dUV[uv_idx + 1] = Vval;
        }
    }
}

// ------------------------------------------------------------------
// Host launcher
// ------------------------------------------------------------------
void launch_draw_box_nv12(
    uint8_t* dY, uint8_t* dUV,
    int W, int H, int pitch,
    int x0, int y0, int w, int h,
    cudaStream_t stream)
{
    static int dbg_calls = 0;

    // clamp ROI dentro al frame per evitare out-of-bounds
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 >= W || y0 >= H) {
        if (dbg_calls < 10) {
            fprintf(stderr,
                    "[draw-box] ROI fuori dal frame: x0=%d y0=%d W=%d H=%d\n",
                    x0, y0, W, H);
            dbg_calls++;
        }
        return;
    }
    if (x0 + w > W) w = W - x0;
    if (y0 + h > H) h = H - y0;
    if (w <= 0 || h <= 0) {
        if (dbg_calls < 10) {
            fprintf(stderr,
                    "[draw-box] ROI degenerata dopo clamp: x0=%d y0=%d w=%d h=%d\n",
                    x0, y0, w, h);
            dbg_calls++;
        }
        return;
    }

    if (dbg_calls < 10) {
        fprintf(stderr,
                "[draw-box] launch W=%d H=%d pitch=%d ROI=(%d,%d,%d,%d) stream=%p\n",
                W, H, pitch, x0, y0, w, h, (void*)stream);
        dbg_calls++;
    }

    dim3 block(32, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    // Colore ben visibile: rosso “vivo” in spazio YUV circa
    const uint8_t Y = 76;   // luma
    const uint8_t U = 84;
    const uint8_t V = 255;

    cudaStream_t s = stream ? stream : 0;

    draw_box_nv12_kernel<<<grid, block, 0, s>>>(
        dY, dUV,
        W, H, pitch,
        x0, y0, w, h,
        Y, U, V
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[draw-box] kernel launch error: %s (W=%d H=%d pitch=%d ROI=(%d,%d,%d,%d))\n",
                cudaGetErrorString(err), W, H, pitch, x0, y0, w, h);
    } else if (dbg_calls < 20) {
        fprintf(stderr, "[draw-box] kernel launched successfully.\n");
        dbg_calls++;
    }
}

} // namespace draw
