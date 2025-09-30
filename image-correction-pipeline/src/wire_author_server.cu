// wire_author_server.cu
#include "wire_author_server.cuh"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// ---- Binary protocol (matches your Python client) ----
static constexpr uint32_t fourcc_u32(const char (&tag)[5]) noexcept {
    return  (uint32_t)(unsigned char)tag[0]
          | ((uint32_t)(unsigned char)tag[1] << 8)
          | ((uint32_t)(unsigned char)tag[2] << 16)
          | ((uint32_t)(unsigned char)tag[3] << 24);
}
static const uint32_t FRAM_MAGIC = fourcc_u32("FRAM");
static const uint32_t SCNT_MAGIC = fourcc_u32("SCNT");
static const uint32_t STMP_MAGIC = fourcc_u32("STMP");

#pragma pack(push,1)
struct FrameHeader { uint32_t magic,w,h,stride,channels,size; };
struct StampsCountHeader { uint32_t magic,count; };
struct Stamp {
    uint32_t magic;
    float    cx, cy;
    float    angle_deg;
    float    sep_px;
    uint32_t bw, bh; // assume bw==bh
};
#pragma pack(pop)

// ---- Tiny TCP helpers ----
static int listen_and_accept(uint16_t port) {
    int s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { perror("socket"); return -1; }
    int yes=1; ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    if (::bind(s, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); ::close(s); return -1; }
    if (::listen(s, 1) < 0) { perror("listen"); ::close(s); return -1; }
    fprintf(stderr, "[wire] listening on %u\n", port);
    sockaddr_in cli{}; socklen_t cl = sizeof(cli);
    int c = ::accept(s, (sockaddr*)&cli, &cl);
    if (c < 0) { perror("accept"); ::close(s); return -1; }
    ::close(s);
    return c;
}
static void send_all(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf; size_t off=0;
    while (off < n) {
        ssize_t k = ::send(fd, p+off, n-off, 0);
        if (k <= 0) { perror("send"); throw std::runtime_error("send"); }
        off += (size_t)k;
    }
}
static void recv_all(int fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf; size_t off=0;
    while (off < n) {
        ssize_t k = ::recv(fd, p+off, n-off, 0);
        if (k <= 0) { perror("recv"); throw std::runtime_error("recv"); }
        off += (size_t)k;
    }
}

// ---- Module state (singleton) ----
namespace {
struct DeviceMask {
    CUdeviceptr dMask = 0;
    size_t      pitch = 0;
    float       dx = 0.f, dy = 0.f;
    bool        valid = false;
};
struct GState {
    // lifetime
    std::atomic<bool> started{false};
    std::thread       th;
    CUcontext         ctx = nullptr;
    uint16_t          port = 0;

    // snapshot handshake
    std::atomic<bool> want_snapshot{false};
    std::atomic<bool> snapshot_ready{false};
    std::mutex              snap_mtx;
    std::condition_variable snap_cv;

    // snapshot buffers
    CUdeviceptr dRGBA = 0; size_t dRGBA_pitch = 0;
    void*   h_rgba = nullptr; size_t h_bytes = 0;
    int snapW = 0, snapH = 0;

    // mask double buffer (inactive->upload, then atomically swap)
    DeviceMask buf[2];
    std::atomic<uint32_t> active{0};
    std::atomic<uint32_t> version{0};
} G;
} // anon

// ---- NV12 → RGBA CUDA kernel (simple BT.601) ----
__global__ void nv12_to_rgba_kernel(
    const uint8_t* __restrict__ Y, int pY,
    const uint8_t* __restrict__ UV, int pUV,
    int W, int H, uint8_t* __restrict__ RGBA, int pRGBA)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=W || y>=H) return;

    int Yv = Y[y*pY + x];
    int cu = x>>1, cv = y>>1;
    const uint8_t* puv = UV + cv*pUV + (cu<<1);
    int U = (int)puv[0] - 128;
    int V = (int)puv[1] - 128;

    float C = (float)Yv - 16.f; if (C<0) C=0;
    float Rf = 1.164f*C + 1.596f*V;
    float Gf = 1.164f*C - 0.392f*U - 0.813f*V;
    float Bf = 1.164f*C + 2.017f*U;

    int R = max(0, min(255, (int)(Rf+0.5f)));
    int Gc = max(0, min(255, (int)(Gf+0.5f)));
    int B = max(0, min(255, (int)(Bf+0.5f)));

    uint8_t* q = RGBA + y*pRGBA + 4*x;
    q[0]=(uint8_t)R; q[1]=(uint8_t)Gc; q[2]=(uint8_t)B; q[3]=255;
}

// ---- Host rasterizer (rotated square OR) ----
static inline int clampi(int v,int a,int b){ return v<a?a:(v>b?b:v); }

static inline void draw_rot_square(std::vector<uint8_t>& M, int W, int H,
                                   float cx, float cy, int side, float angle_deg, float pad_px)
{
    const float half = 0.5f*(float)side + pad_px;
    const float th = angle_deg * (float)M_PI / 180.f;
    const float c = cosf(th), s = sinf(th);
    const float r00=c, r01=s, r10=-s, r11=c;
    const float ext = half*(fabsf(c)+fabsf(s));
    int x0 = clampi((int)floorf(cx - ext), 0, W-1);
    int x1 = clampi((int)ceilf (cx + ext)+1, 1, W);
    int y0 = clampi((int)floorf(cy - ext), 0, H-1);
    int y1 = clampi((int)ceilf (cy + ext)+1, 1, H);
    for (int y=y0; y<y1; ++y) {
        uint8_t* row = M.data() + (size_t)y*W;
        for (int x=x0; x<x1; ++x) {
            float dx = (float)x + 0.5f - cx;
            float dy = (float)y + 0.5f - cy;
            float lx = r00*dx + r01*dy;
            float ly = r10*dx + r11*dy;
            if (fabsf(lx) <= half && fabsf(ly) <= half) row[x] = 255;
        }
    }
}

// Build (OR of all squares), upload to inactive buffer, swap active.
static int build_and_swap_mask_from_stamps(const std::vector<Stamp>& stamps)
{
    if (G.snapW<=0 || G.snapH<=0) return -1;
    const int W = G.snapW, H = G.snapH;

    if (stamps.empty()) return -1;

    // Donor vector from first stamp (stored for later use/debug)
    float angle = stamps.front().angle_deg;
    float sep   = stamps.front().sep_px;
    float ang = angle * (float)M_PI / 180.f;
    float vx = cosf(ang), vy = sinf(ang);
    float dx = -vx * (sep*0.5f);
    float dy = -vy * (sep*0.5f);

    // Rasterize into a single 8-bit mask
    std::vector<uint8_t> hostMask((size_t)W*(size_t)H, 0u);
    const float PAD = 0.75f; // tiny overlap to hide seams
    for (const auto& s : stamps) {
        float cx = std::min((float)(W-1), std::max(0.f, s.cx));
        float cy = std::min((float)(H-1), std::max(0.f, s.cy));
        int side = (int)s.bw;
        draw_rot_square(hostMask, W, H, cx, cy, side, s.angle_deg, PAD);
    }

    // Upload to the inactive slot
    uint32_t act = G.active.load(std::memory_order_acquire);
    uint32_t ina = act ^ 1;
    DeviceMask& dm = G.buf[ina];

    if (!dm.dMask) {
        CUresult rc = cuMemAllocPitch(&dm.dMask, &dm.pitch, (size_t)W, (size_t)H, 4);
        if (rc != CUDA_SUCCESS) { fprintf(stderr, "[wire] cuMemAllocPitch(mask) failed\n"); return -1; }
    }

    CUDA_MEMCPY2D c{};
    c.srcMemoryType = CU_MEMORYTYPE_HOST;   c.srcHost   = hostMask.data(); c.srcPitch = (size_t)W;
    c.dstMemoryType = CU_MEMORYTYPE_DEVICE; c.dstDevice = dm.dMask;        c.dstPitch = dm.pitch;
    c.WidthInBytes  = (size_t)W; c.Height = (size_t)H;
    cuMemcpy2D(&c);

    dm.dx = dx; dm.dy = dy; dm.valid = true;

    // Atomically swap active
    G.active.store(ina, std::memory_order_release);
    uint32_t ver = G.version.fetch_add(1, std::memory_order_acq_rel) + 1;
    fprintf(stderr, "[wire] mask uploaded to device (W=%d H=%d), version=%u, active=%u, dx=%.2f dy=%.2f\n",
            W, H, ver, ina, dx, dy);
    return W*H;
}

// ---- Dedicated network thread (one-shot authoring round) ----
static void net_thread_body()
{
    cuCtxSetCurrent(G.ctx);

    try {
        int cfd = listen_and_accept(G.port);
        if (cfd < 0) return;

        // Request a snapshot from the next frame
        G.want_snapshot = true;
        {
            std::unique_lock<std::mutex> lk(G.snap_mtx);
            G.snap_cv.wait(lk, []{ return G.snapshot_ready.load(); });
        }

        // Send FRAM + RGBA to the client
        FrameHeader fh{};
        fh.magic=FRAM_MAGIC; fh.w=G.snapW; fh.h=G.snapH;
        fh.stride=(uint32_t)G.snapW*4; fh.channels=4; fh.size=(uint32_t)(G.snapW*G.snapH*4);
        send_all(cfd, &fh, sizeof(fh));
        send_all(cfd, G.h_rgba, fh.size);
        fprintf(stderr, "[wire] sent snapshot %dx%d RGBA\n", G.snapW, G.snapH);

        // Receive SCNT + STMP×N
        StampsCountHeader sc{}; recv_all(cfd, &sc, sizeof(sc));
        if (sc.magic != SCNT_MAGIC) throw std::runtime_error("SCNT magic mismatch");
        std::vector<Stamp> stamps(sc.count);
        for (uint32_t i=0;i<sc.count;i++) {
            recv_all(cfd, &stamps[i], sizeof(Stamp));
            if (stamps[i].magic != STMP_MAGIC) throw std::runtime_error("STMP magic mismatch");
        }
        ::close(cfd);
        fprintf(stderr, "[wire] received %u stamp(s) from client\n", sc.count);

        // Build mask + upload to device memory (not used by any kernel yet)
        build_and_swap_mask_from_stamps(stamps);
    } catch (const std::exception& e) {
        fprintf(stderr, "[wire] net thread error: %s\n", e.what());
    }
}

// ---- Public API ----
void wire_author_start(CUcontext ctx, uint16_t tcp_port)
{
    bool expected = false;
    if (!G.started.compare_exchange_strong(expected, true)) return; // already started

    // If caller didn't pass a context, retain CUDA primary here.
    if (!ctx) {
        cuInit(0);
        CUdevice dev = 0;
        cuDeviceGet(&dev, 0);
        cuDevicePrimaryCtxRetain(&ctx, dev);
    }

    G.ctx  = ctx;
    G.port = tcp_port;
    G.want_snapshot = false;
    G.snapshot_ready = false;
    G.th = std::thread(net_thread_body);
}

void wire_author_stop()
{
    if (G.started.exchange(false)) {
        if (G.th.joinable()) G.th.join();

        // Free device/host buffers
        for (int i=0;i<2;i++) {
            if (G.buf[i].dMask) { cuMemFree(G.buf[i].dMask); G.buf[i].dMask = 0; G.buf[i].pitch = 0; G.buf[i].valid=false; }
        }
        if (G.dRGBA) { cuMemFree(G.dRGBA); G.dRGBA = 0; G.dRGBA_pitch = 0; }
        if (G.h_rgba) { cudaFreeHost(G.h_rgba); G.h_rgba = nullptr; G.h_bytes = 0; }
        G.snapW=G.snapH=0;
        fprintf(stderr, "[wire] authoring helper stopped\n");
    }
}

void wire_author_snapshot_nv12_if_needed(
    const uint8_t* dY, int pitchY,
    const uint8_t* dUV, int pitchUV,
    int W, int H, cudaStream_t stream)
{
    if (!G.want_snapshot.exchange(false)) return; // no request pending

    // Allocate device RGBA if needed
    if (!G.dRGBA || G.dRGBA_pitch < (size_t)W*4) {
        if (G.dRGBA) cuMemFree(G.dRGBA);
        cuMemAllocPitch(&G.dRGBA, &G.dRGBA_pitch, (size_t)W*4, (size_t)H, 16);
    }
    // Allocate pinned host RGBA if needed
    size_t need = (size_t)W * (size_t)H * 4;
    if (!G.h_rgba || G.h_bytes != need) {
        if (G.h_rgba) cudaFreeHost(G.h_rgba);
        cudaHostAlloc(&G.h_rgba, need, cudaHostAllocDefault);
        G.h_bytes = need;
    }

    // Convert NV12 -> RGBA on GPU then copy to pinned host
    dim3 b(16,16), g((W+b.x-1)/b.x, (H+b.y-1)/b.y);
    nv12_to_rgba_kernel<<<g,b,0,stream>>>(
        dY, pitchY, dUV, pitchUV, W, H,
        (uint8_t*)(uintptr_t)G.dRGBA, (int)G.dRGBA_pitch);

    CUDA_MEMCPY2D c{};
    c.srcMemoryType=CU_MEMORYTYPE_DEVICE; c.srcDevice=G.dRGBA; c.srcPitch=G.dRGBA_pitch;
    c.dstMemoryType=CU_MEMORYTYPE_HOST;   c.dstHost=G.h_rgba;  c.dstPitch=(size_t)W*4;
    c.WidthInBytes=(size_t)W*4; c.Height=(size_t)H;
    cuMemcpy2DAsync(&c, (CUstream)stream);
    cudaStreamSynchronize(stream);

    G.snapW = W; G.snapH = H;
    G.snapshot_ready = true;
    G.snap_cv.notify_all();
    fprintf(stderr, "[wire] snapshot ready (%dx%d)\n", W, H);
}

bool wire_author_get_active_mask(
    CUdeviceptr* dMask, int* pitchBytes,
    float* dx, float* dy,
    uint32_t* version)
{
    uint32_t a = G.active.load(std::memory_order_acquire);
    const DeviceMask& dm = G.buf[a];
    if (!dm.valid || !dm.dMask) return false;
    if (dMask)      *dMask = dm.dMask;
    if (pitchBytes) *pitchBytes = (int)dm.pitch;
    if (dx)         *dx = dm.dx;
    if (dy)         *dy = dm.dy;
    if (version)    *version = G.version.load(std::memory_order_relaxed);
    return true;
}
