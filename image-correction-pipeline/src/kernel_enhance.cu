/**
 * @file kernel_enhance.cu
 * @brief CUDA kernels for image enhancement (NV12 frames).
 *
 * Implements:
 *  - Gamma correction
 *  - Gaussian blur (for tone mapping + sharpening)
 *  - Local tone mapping
 *  - Soft-limited unsharp masking
 *  - Adaptive saturation
 *  - Exposure guard via histogram percentile
 *  - Highlight rolloff (for bright sky regions)
 *
 * Each function has a launcher that operates on NV12 planes.
 */
#include "kernel_enhance.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace icp {

template<typename T> static inline T divUp(T a, T b){ return (a+b-1)/b; }
__device__ __forceinline__ uint8_t clamp_u8f(float v){ v = fminf(fmaxf(v,0.f),255.f); return (uint8_t)(v+0.5f); }

// --- Gamma ---
__global__ void gammaY(uint8_t* Y,int W,int H,int pitch,float gamma){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    float v=Y[y*pitch+x]/255.f;
    v=powf(v, fmaxf(gamma,1e-4f));
    Y[y*pitch+x]=clamp_u8f(v*255.f);
}

// --- Gauss blur separabile ---
__constant__ float kG[5]={0.0625f,0.25f,0.375f,0.25f,0.0625f};
__global__ void gauss5x1_h(const uint8_t* inY,uint8_t* tmp,int W,int H,int pitch){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    float acc=0.f;
    #pragma unroll
    for(int i=-2;i<=2;i++){
        int xx=min(max(x+i,0),W-1);
        acc+=kG[i+2]*inY[y*pitch+xx];
    }
    tmp[y*pitch+x]=(uint8_t)(acc+0.5f);
}
__global__ void gauss1x5_v(const uint8_t* tmp,uint8_t* outY,int W,int H,int pitch){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    float acc=0.f;
    #pragma unroll
    for(int j=-2;j<=2;j++){ int yy=min(max(y+j,0),H-1); acc+=kG[j+2]*tmp[yy*pitch+x]; }
    outY[y*pitch+x]=(uint8_t)(acc+0.5f);
}

// --- Local tone mapping ---
__global__ void toneMapLocal(uint8_t* Y,const uint8_t* Yb,int W,int H,int pitch,float a,float Lwhite){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    float yl=Y[y*pitch+x]/255.f, L=Yb[y*pitch+x]/255.f;
    float num=yl*(1.f+yl/(Lwhite*Lwhite));
    float den=1.f+a*L;
    float out=num/fmaxf(den,1e-4f);
    Y[y*pitch+x]=clamp_u8f(fminf(out,1.f)*255.f);
}

// --- Sharpen (soft-limited) ---
__global__ void unsharp_limited(uint8_t* Y,const uint8_t* Yb,int W,int H,int pitch,float amount,float clip){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    float y0=Y[y*pitch+x], ybl=Yb[y*pitch+x];
    float d=y0-ybl;
    float lim=clip<1.f?1.f:clip;
    float d_lim=tanhf(d/lim)*lim;
    float out=y0+amount*d_lim;
    Y[y*pitch+x]=clamp_u8f(out);
}

// --- Saturation adaptive ---
__global__ void saturationUV_adaptive(uint8_t* UV,const uint8_t* Y,
                                      int W,int H,int pitchUV,int pitchY,float s){
    int xPair=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(xPair>=W/2||y>=H/2) return;
    int offUV=y*pitchUV+(xPair<<1);

    int x=xPair<<1, yy=y<<1;
    float y00=Y[(yy  )*pitchY+x];
    float y01=Y[(yy  )*pitchY+x+1];
    float y10=Y[(yy+1)*pitchY+x];
    float y11=Y[(yy+1)*pitchY+x+1];
    float yAvg=0.25f*(y00+y01+y10+y11)/255.f;

    float t=fminf(fmaxf((yAvg-0.85f)/0.15f,0.f),1.f);
    float s_eff=(1.f-t)*s+t*0.92f;

    float U=UV[offUV+0]-128.f, V=UV[offUV+1]-128.f;
    U*=s_eff; V*=s_eff;
    UV[offUV+0]=clamp_u8f(U+128.f);
    UV[offUV+1]=clamp_u8f(V+128.f);
}

// --- Hist only ROI (ignora top 25%) ---
__global__ void histY256_roi(const uint8_t* Y,int W,int H,int pitch,unsigned int* gHist,int y_start){
    __shared__ unsigned int s[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x) s[i]=0; __syncthreads();
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    for(int y=y_start+blockIdx.y*blockDim.y;y<H;y+=gridDim.y*blockDim.y){
        if(x<W){ unsigned int v=Y[y*pitch+x]; atomicAdd(&s[v],1); }
    }
    __syncthreads();
    for(int i=threadIdx.x;i<256;i+=blockDim.x) atomicAdd(&gHist[i],s[i]);
}
__global__ void mulGainClampY(uint8_t* Y,int W,int H,int pitch,float g){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    float v=Y[y*pitch+x]*g;
    Y[y*pitch+x]=clamp_u8f(v);
}

// --- Rolloff top-weighted ---
__global__ void rolloffY_top(uint8_t* Y,int W,int H,int pitch,float startN,float ymaxN,float strength){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    float yN=Y[y*pitch+x]/255.f;
    if(yN>startN){
        float over=(yN-startN)/(1.f-startN);
        float wy=0.7f+0.3f*(1.f-(float)y/(H-1));
        float c=(1.f-expf(-(strength*wy)*over));
        yN=startN+c*(ymaxN-startN);
    }
    Y[y*pitch+x]=(uint8_t)(fminf(fmaxf(yN,0.f),1.f)*255.f+0.5f);
}

// --- Launchers ---
void launch_enhance_nv12(
    uint8_t* dY,int W,int H,int pitchY,
    uint8_t* dUV,int pitchUV,
    const EnhanceParams& p,
    cudaStream_t stream)
{
    dim3 block(16,16);
    dim3 gridY(divUp(W,16),divUp(H,16));
    dim3 gridUV(divUp(W/2,16),divUp(H/2,16));

    if(fabsf(p.gamma-1.f)>1e-3f)
        gammaY<<<gridY,block,0,stream>>>(dY,W,H,pitchY,p.gamma);

    size_t pitchTmp;
    uint8_t *tmp=nullptr,*yb=nullptr;
    cudaMallocPitch(&tmp,&pitchTmp,W,H);
    cudaMallocPitch(&yb,&pitchTmp,W,H);

    gauss5x1_h<<<gridY,block,0,stream>>>(dY,tmp,W,H,pitchY);
    gauss1x5_v<<<gridY,block,0,stream>>>(tmp,yb,W,H,pitchY);

    if(p.local_tm>1e-3f)
        toneMapLocal<<<gridY,block,0,stream>>>(dY,yb,W,H,pitchY,p.local_tm,p.tm_white);
    if(p.sharpen_amount>1e-3f)
        unsharp_limited<<<gridY,block,0,stream>>>(dY,yb,W,H,pitchY,p.sharpen_amount,p.sharpen_clip);

    cudaFree(tmp); cudaFree(yb);

    if(fabsf(p.saturation-1.f)>1e-3f)
        saturationUV_adaptive<<<gridUV,block,0,stream>>>(dUV,dY,W,H,pitchUV,pitchY,p.saturation);
}

void launch_highlight_guard(uint8_t* dY,int W,int H,int pitchY,float* pGain,cudaStream_t stream){
    const float PCTL=0.99f;
    const float TARGET=215.f;
    const float SMOOTH=0.25f;
    const float MAX_UP=1.00f, MAX_DOWN=0.45f;

    unsigned int* dHist=nullptr; cudaMalloc(&dHist,256*sizeof(unsigned int));
    cudaMemsetAsync(dHist,0,256*sizeof(unsigned int),stream);

    dim3 b(256,1), g((W+255)/256,min((H+15)/16,64));
    int y_start=(int)(0.25f*H);
    histY256_roi<<<g,b,0,stream>>>(dY,W,H,pitchY,dHist,y_start);

    unsigned int hHist[256]; cudaMemcpyAsync(hHist,dHist,256*sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    cudaFree(dHist);

    unsigned long long tot=(unsigned long long)W*H*3/4;
    unsigned long long thr=(unsigned long long)(PCTL*tot);
    unsigned long long acc=0; int p99=255;
    for(int i=0;i<256;i++){ acc+=hHist[i]; if(acc>=thr){ p99=i; break; } }

    float targetGain=1.0f;
    if(p99>(int)TARGET) targetGain=TARGET/(float)p99;
    targetGain=fminf(fmaxf(targetGain,MAX_DOWN),MAX_UP);
    *pGain=(1.f-SMOOTH)*(*pGain)+SMOOTH*targetGain;

    dim3 block(16,16), grid(divUp(W,16),divUp(H,16));
    mulGainClampY<<<grid,block,0,stream>>>(dY,W,H,pitchY,*pGain);
}

void launch_highlight_rolloff_top(uint8_t* dY,int W,int H,int pitchY,float startN,float ymaxN,float strength,cudaStream_t stream){
    dim3 b(16,16), g(divUp(W,16),divUp(H,16));
    rolloffY_top<<<g,b,0,stream>>>(dY,W,H,pitchY,startN,ymaxN,strength);
}

} // namespace icp
