#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

namespace draw {

void launch_draw_box_nv12(
    uint8_t* dY, uint8_t* dUV,
    int W, int H, int pitch,
    int x0, int y0, int w, int h,
    cudaStream_t stream);

} // namespace draw
