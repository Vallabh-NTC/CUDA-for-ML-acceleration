// overlay.cu — flow field arrows + resultant vector on NV12 EGLImage

#include "overlay.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ── Pixel helpers ─────────────────────────────────────────────────────────────
__device__ inline void put_y(
    uint8_t *d_y, int pitch, int W, int H, int x, int y, uint8_t v)
{
    if ((unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H)
        d_y[y * pitch + x] = v;
}

__device__ inline void put_uv(
    uint8_t *d_uv, int pitch, int W, int H, int x, int y, uint8_t U, uint8_t V)
{
    int ux=x>>1, uy=y>>1;
    if ((unsigned)ux<(unsigned)(W>>1) && (unsigned)uy<(unsigned)(H>>1)) {
        d_uv[uy*pitch + ux*2+0] = U;
        d_uv[uy*pitch + ux*2+1] = V;
    }
}

__device__ void draw_line(
    uint8_t *d_y, uint8_t *d_uv,
    int pitchY, int pitchUV, int W, int H,
    int x0, int y0, int x1, int y1,
    uint8_t Yc, uint8_t Uc, uint8_t Vc)
{
    int dx=abs(x1-x0), sx=x0<x1?1:-1;
    int dy=-abs(y1-y0), sy=y0<y1?1:-1;
    int err=dx+dy;
    for (int i=0;i<1024;++i) {
        put_y (d_y,  pitchY,  W,H,x0,y0,Yc);
        put_uv(d_uv, pitchUV, W,H,x0,y0,Uc,Vc);
        if (x0==x1&&y0==y1) break;
        int e2=2*err;
        if (e2>=dy){err+=dy;x0+=sx;}
        if (e2<=dx){err+=dx;y0+=sy;}
    }
}

__device__ void draw_arrow(
    uint8_t *d_y, uint8_t *d_uv,
    int pitchY, int pitchUV, int W, int H,
    int x0, int y0, int x1, int y1,
    uint8_t Yc, uint8_t Uc, uint8_t Vc,
    int thickness=1)
{
    for (int oy=-thickness/2; oy<=thickness/2; ++oy)
    for (int ox=-thickness/2; ox<=thickness/2; ++ox) {
        draw_line(d_y,d_uv,pitchY,pitchUV,W,H,
                  x0+ox,y0+oy,x1+ox,y1+oy,Yc,Uc,Vc);
    }
    // Arrow head
    int hx=x1-x0, hy=y1-y0;
    int len=max(1,abs(hx)+abs(hy));
    const int HL=10, HW=6;
    int ahx=(hx*HL)/len, ahy=(hy*HL)/len;
    int apx=(-hy*HW)/len, apy=(hx*HW)/len;
    draw_line(d_y,d_uv,pitchY,pitchUV,W,H, x1,y1, x1-ahx+apx,y1-ahy+apy, Yc,Uc,Vc);
    draw_line(d_y,d_uv,pitchY,pitchUV,W,H, x1,y1, x1-ahx-apx,y1-ahy-apy, Yc,Uc,Vc);
}

// ── Flow field kernel ─────────────────────────────────────────────────────────
__global__ void overlay_field_kernel(
    uint8_t     *d_y, uint8_t *d_uv,
    int          pitchY, int pitchUV, int W, int H,
    const float *d_flow,
    int roi_x0, int roi_x1, int roi_y0, int roi_y1,
    int step, float arrow_scale, float min_mag)
{
    const int sx=blockIdx.x*blockDim.x+threadIdx.x;
    const int sy=blockIdx.y*blockDim.y+threadIdx.y;
    const int px=roi_x0+sx*step;
    const int py=roi_y0+sy*step;
    if (px>=roi_x1||py>=roi_y1||px<0||py<0||px>=W||py>=H) return;

    const int plane=H*W, idx=py*W+px;
    const float u=d_flow[0*plane+idx];
    const float v=d_flow[1*plane+idx];
    if (sqrtf(u*u+v*v)<min_mag) return;

    int x1=max(0,min(W-1,px+(int)lrintf(u*arrow_scale)));
    int y1=max(0,min(H-1,py+(int)lrintf(v*arrow_scale)));

    // Green arrows for field (Y=150, U=44, V=21 → greenish)
    draw_arrow(d_y,d_uv,pitchY,pitchUV,W,H, px,py,x1,y1, 150,44,21, 1);
}

// ── Resultant vector kernel (single thread) ───────────────────────────────────
// Blue arrow in NV12: Y=29, U=255, V=107
__global__ void overlay_resultant_kernel(
    uint8_t    *d_y, uint8_t *d_uv,
    int         pitchY, int pitchUV, int W, int H,
    float       mean_u, float mean_v,
    int         cx, int cy,          // center of ROI in pixels
    float       result_scale)        // visual scale for resultant
{
    if (threadIdx.x!=0||blockIdx.x!=0) return;

    int x1=max(0,min(W-1, cx+(int)lrintf(mean_u*result_scale)));
    int y1=max(0,min(H-1, cy+(int)lrintf(mean_v*result_scale)));

    // Draw thick blue arrow (thickness=4)
    draw_arrow(d_y,d_uv,pitchY,pitchUV,W,H, cx,cy,x1,y1,
               29,255,107, 4);
}

// ── Host wrappers ─────────────────────────────────────────────────────────────
void overlay_draw_flow(
    uint8_t     *d_y, uint8_t *d_uv,
    int          pitchY, int pitchUV, int W, int H,
    const float *d_flow,
    float roi_x0, float roi_x1, float roi_y0, float roi_y1,
    int step, float arrow_scale, float min_mag,
    cudaStream_t stream)
{
    const int rx0=(int)(roi_x0*W), rx1=(int)(roi_x1*W);
    const int ry0=(int)(roi_y0*H), ry1=(int)(roi_y1*H);
    const int nx=(rx1-rx0+step-1)/step, ny=(ry1-ry0+step-1)/step;
    dim3 block(8,8);
    dim3 grid((nx+7)/8,(ny+7)/8);
    overlay_field_kernel<<<grid,block,0,stream>>>(
        d_y,d_uv,pitchY,pitchUV,W,H,d_flow,
        rx0,rx1,ry0,ry1,step,arrow_scale,min_mag);
}

void overlay_draw_resultant(
    uint8_t    *d_y, uint8_t *d_uv,
    int         pitchY, int pitchUV, int W, int H,
    float       mean_u, float mean_v,
    float       roi_x0, float roi_x1,
    float       roi_y0, float roi_y1,
    float       result_scale,
    cudaStream_t stream)
{
    // Center of ROI
    const int cx = (int)((roi_x0 + roi_x1) * 0.5f * W);
    const int cy = (int)((roi_y0 + roi_y1) * 0.5f * H);

    overlay_resultant_kernel<<<1,1,0,stream>>>(
        d_y,d_uv,pitchY,pitchUV,W,H,
        mean_u, mean_v, cx, cy, result_scale);
}