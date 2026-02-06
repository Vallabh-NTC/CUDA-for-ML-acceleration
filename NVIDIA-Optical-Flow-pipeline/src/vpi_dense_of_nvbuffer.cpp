// vpi_gst_of_rtp_overlay_levelA_nvmm_nvbuffer.cpp
//
// LEVEL A (Jetson Orin / JetPack 5.1.4 / VPI 2.4.8):
// MP4(H264) -> nvv4l2decoder -> NVMM (DMABUF fd) -> VPI wrapper using NVBUFFER (ZERO-COPY input)
// -> VPI VIC convert (NV12 -> Y8) -> CUDA pyramid + VIC tiling + OFA
// -> draw arrows (CPU, no OpenCV) + resultant vector (CPU; base frame built from Y8 readback)
// -> RTP/H264 UDP out (software x264enc, same as your original)
//
// NOTES:
// - Input is zero-copy into VPI (decoder produces NVMM; VPI wraps the NvBuffer fd).
// - Overlay and appsrc output still copy on CPU (not end-to-end zero-copy).
// - Texture gate is disabled in Level A (we no longer read luma directly from appsink CPU pointer).
//
// Receiver example:
// gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
//   ! rtpjitterbuffer ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
//
// Build example:
// g++ -O3 -DNDEBUG -std=gnu++17 vpi_gst_of_rtp_overlay_levelA_nvmm_nvbuffer.cpp -o vpi_levelA \
//   `pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0` \
//   -lvpi
//
// Run example:
// ./vpi_levelA /home/ntc-orin/Videos/StraightandBack_car_movement_always10kmph.mp4 2560 720

#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Pyramid.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <gst/allocators/gstdmabuf.h>

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK_VPI(stmt)                                                         \
    do {                                                                        \
        VPIStatus _st = (stmt);                                                 \
        if (_st != VPI_SUCCESS) {                                               \
            char msg[512];                                                      \
            vpiGetLastStatusMessage(msg, sizeof(msg));                          \
            std::cerr << "VPI error: " << vpiStatusGetName(_st)                 \
                      << " at " << #stmt << "\n"                                \
                      << "Message: " << msg << "\n";                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

struct VPIImageLockGuard {
    VPIImage img = nullptr;
    explicit VPIImageLockGuard(VPIImage i) : img(i) {}
    ~VPIImageLockGuard() { if (img) vpiImageUnlock(img); }
    VPIImageLockGuard(const VPIImageLockGuard&) = delete;
    VPIImageLockGuard& operator=(const VPIImageLockGuard&) = delete;
};

static inline float s10_5_to_px(int16_t v) { return float(v) / 32.0f; }

// ---------------- Drawing (no OpenCV) ----------------

static inline void put_px_rgba(uint8_t *img, int W, int H, int x, int y,
                               uint8_t r, uint8_t g, uint8_t b, uint8_t a=255)
{
    if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return;
    uint8_t *p = img + (size_t)(y * W + x) * 4;
    p[0] = r; p[1] = g; p[2] = b; p[3] = a;
}

static void draw_line_rgba(uint8_t *img, int W, int H,
                           int x0, int y0, int x1, int y1,
                           uint8_t r, uint8_t g, uint8_t b, int thickness)
{
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    while (true) {
        for (int oy = -thickness/2; oy <= thickness/2; ++oy)
            for (int ox = -thickness/2; ox <= thickness/2; ++ox)
                put_px_rgba(img, W, H, x0 + ox, y0 + oy, r, g, b, 255);

        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

static void draw_arrow_rgba_color(uint8_t *img, int W, int H,
                                  int x0, int y0, int x1, int y1,
                                  int thickness,
                                  uint8_t innerR, uint8_t innerG, uint8_t innerB,
                                  uint8_t outlineR=0, uint8_t outlineG=0, uint8_t outlineB=0)
{
    draw_line_rgba(img, W, H, x0, y0, x1, y1, outlineR, outlineG, outlineB, thickness+2);
    draw_line_rgba(img, W, H, x0, y0, x1, y1, innerR, innerG, innerB, thickness);

    float dx = float(x1 - x0);
    float dy = float(y1 - y0);
    float L  = std::sqrt(dx*dx + dy*dy);
    if (L < 1.0f) return;

    float ux = dx / L, uy = dy / L;
    float headLen = std::max(6.0f, float(thickness) * 3.0f);
    float ang = 0.6f; // ~34 deg
    float cx = float(x1), cy = float(y1);

    float vx1 =  std::cos(ang)*ux - std::sin(ang)*uy;
    float vy1 =  std::sin(ang)*ux + std::cos(ang)*uy;
    float vx2 =  std::cos(-ang)*ux - std::sin(-ang)*uy;
    float vy2 =  std::sin(-ang)*ux + std::cos(-ang)*uy;

    int hx1 = int(std::lround(cx - headLen * vx1));
    int hy1 = int(std::lround(cy - headLen * vy1));
    int hx2 = int(std::lround(cx - headLen * vx2));
    int hy2 = int(std::lround(cy - headLen * vy2));

    draw_line_rgba(img, W, H, x1, y1, hx1, hy1, outlineR, outlineG, outlineB, thickness+2);
    draw_line_rgba(img, W, H, x1, y1, hx2, hy2, outlineR, outlineG, outlineB, thickness+2);
    draw_line_rgba(img, W, H, x1, y1, hx1, hy1, innerR, innerG, innerB, thickness);
    draw_line_rgba(img, W, H, x1, y1, hx2, hy2, innerR, innerG, innerB, thickness);
}

static void draw_arrow_rgba(uint8_t *img, int W, int H,
                            int x0, int y0, int x1, int y1,
                            int thickness)
{
    draw_arrow_rgba_color(img, W, H, x0, y0, x1, y1, thickness, 0,255,0, 0,0,0);
}

// Flow grid contains float2 per cell (vx,vy) in pixels.
struct F2 { float x,y; };

static inline F2 bilinear_flow(const F2* grid, int mvW, int mvH, float gx, float gy)
{
    gx = std::clamp(gx, 0.0f, float(mvW - 1));
    gy = std::clamp(gy, 0.0f, float(mvH - 1));
    int x0 = (int)std::floor(gx), y0 = (int)std::floor(gy);
    int x1 = std::min(x0 + 1, mvW - 1);
    int y1 = std::min(y0 + 1, mvH - 1);
    float tx = gx - x0;
    float ty = gy - y0;

    auto at = [&](int x, int y)->F2 { return grid[y*mvW + x]; };

    F2 a = at(x0,y0), b = at(x1,y0), c = at(x0,y1), d = at(x1,y1);
    F2 ab{ a.x + tx*(b.x - a.x), a.y + tx*(b.y - a.y) };
    F2 cd{ c.x + tx*(d.x - c.x), c.y + tx*(d.y - c.y) };
    return F2{ ab.x + ty*(cd.x - ab.x), ab.y + ty*(cd.y - ab.y) };
}

// Convert GRAY8 (with stride) to RGBA, used as the base image for overlay.
static void gray_to_rgba_stride(std::vector<uint8_t>& rgba,
                               const uint8_t* gray, int stride,
                               int W, int H)
{
    rgba.resize((size_t)W * (size_t)H * 4);
    uint8_t* out = rgba.data();

    for (int y = 0; y < H; ++y) {
        const uint8_t* row = gray + (size_t)y * (size_t)stride;
        for (int x = 0; x < W; ++x) {
            uint8_t v = row[x];
            size_t i = ((size_t)y * (size_t)W + (size_t)x) * 4;
            out[i+0] = v;
            out[i+1] = v;
            out[i+2] = v;
            out[i+3] = 255;
        }
    }
}

// ---------------- Stats helpers ----------------

static inline bool inb(int x,int y,int W,int H)
{
    return (unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H;
}

static float percentile_inplace(std::vector<float>& v, float p01)
{
    if (v.empty()) return 0.0f;
    p01 = std::clamp(p01, 0.0f, 1.0f);
    size_t k = (size_t)std::lround(p01 * float(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t)k, v.end());
    return v[k];
}

static F2 median_vec_2d(std::vector<F2>& neigh)
{
    std::vector<float> xs, ys;
    xs.reserve(neigh.size());
    ys.reserve(neigh.size());
    for (auto &v : neigh) { xs.push_back(v.x); ys.push_back(v.y); }

    auto med = [](std::vector<float>& a)->float {
        size_t k = a.size()/2;
        std::nth_element(a.begin(), a.begin() + (ptrdiff_t)k, a.end());
        return a[k];
    };

    return F2{ med(xs), med(ys) };
}

// ---------------- Draw arrows (Level A: no texture gate) ----------------

static void draw_big_green_arrows_rgba_no_texture(
    uint8_t *rgba, int W, int H,
    const F2* flowGrid, int mvW, int mvH,
    int gridStep,
    int stepPx,
    float scale,
    float maxLenPx,
    float magThresholdMin,
    int thickness,
    bool  enableDynamicThreshold,
    float dynamicP75Factor,
    bool  enableNeighborGate,
    float neighborDiffThr
)
{
    float magThr = magThresholdMin;

    if (enableDynamicThreshold) {
        std::vector<float> mags;
        mags.reserve((size_t)(W/stepPx + 2) * (size_t)(H/stepPx + 2));

        for (int y = 0; y < H; y += stepPx) {
            for (int x = 0; x < W; x += stepPx) {
                float gx = float(x) / float(gridStep);
                float gy = float(y) / float(gridStep);
                F2 f = bilinear_flow(flowGrid, mvW, mvH, gx, gy);
                mags.push_back(std::sqrt(f.x*f.x + f.y*f.y));
            }
        }

        float p75 = percentile_inplace(mags, 0.75f);
        magThr = std::max(magThresholdMin, p75 * dynamicP75Factor);
    }

    std::vector<F2> neigh; neigh.reserve(9);

    for (int y = 0; y < H; y += stepPx) {
        for (int x = 0; x < W; x += stepPx) {

            float gx = float(x) / float(gridStep);
            float gy = float(y) / float(gridStep);
            F2 f = bilinear_flow(flowGrid, mvW, mvH, gx, gy);

            float mag = std::sqrt(f.x*f.x + f.y*f.y);
            if (mag < magThr) continue;

            if (enableNeighborGate) {
                int ix = (int)std::lround(gx);
                int iy = (int)std::lround(gy);
                ix = std::clamp(ix, 0, mvW-1);
                iy = std::clamp(iy, 0, mvH-1);

                neigh.clear();
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        int nx = ix + ox, ny = iy + oy;
                        if (inb(nx, ny, mvW, mvH)) neigh.push_back(flowGrid[ny*mvW + nx]);
                    }
                }

                if (neigh.size() >= 5) {
                    F2 m = median_vec_2d(neigh);
                    float diff = std::hypot(f.x - m.x, f.y - m.y);
                    if (diff > neighborDiffThr) continue;
                }
            }

            float dx = f.x * scale;
            float dy = f.y * scale;

            float L = std::sqrt(dx*dx + dy*dy);
            if (L > 1e-6f && L > maxLenPx) {
                float s = maxLenPx / L;
                dx *= s; dy *= s;
            }

            int x1 = (int)std::lround(float(x) + dx);
            int y1 = (int)std::lround(float(y) + dy);
            draw_arrow_rgba(rgba, W, H, x, y, x1, y1, thickness);
        }
    }
}

// ---------------- Resultant vector (Level A: no texture gate) ----------------

struct ResultantFlow {
    F2 sum;
    F2 avg;
    int count;
};

static ResultantFlow compute_resultant_flow_no_texture(
    int W, int H,
    const F2* flowGrid, int mvW, int mvH,
    int gridStep,
    int stepPx,
    float magThresholdMin,
    bool  enableDynamicThreshold,
    float dynamicP75Factor,
    bool  enableNeighborGate,
    float neighborDiffThr
)
{
    float magThr = magThresholdMin;

    if (enableDynamicThreshold) {
        std::vector<float> mags;
        mags.reserve((size_t)(W/stepPx + 2) * (size_t)(H/stepPx + 2));

        for (int y = 0; y < H; y += stepPx) {
            for (int x = 0; x < W; x += stepPx) {
                float gx = float(x) / float(gridStep);
                float gy = float(y) / float(gridStep);
                F2 f = bilinear_flow(flowGrid, mvW, mvH, gx, gy);
                mags.push_back(std::sqrt(f.x*f.x + f.y*f.y));
            }
        }

        float p75 = percentile_inplace(mags, 0.75f);
        magThr = std::max(magThresholdMin, p75 * dynamicP75Factor);
    }

    std::vector<F2> neigh; neigh.reserve(9);

    double sumX = 0.0;
    double sumY = 0.0;
    int count = 0;

    for (int y = 0; y < H; y += stepPx) {
        for (int x = 0; x < W; x += stepPx) {

            float gx = float(x) / float(gridStep);
            float gy = float(y) / float(gridStep);
            F2 f = bilinear_flow(flowGrid, mvW, mvH, gx, gy);

            float mag = std::sqrt(f.x*f.x + f.y*f.y);
            if (mag < magThr) continue;

            if (enableNeighborGate) {
                int ix = (int)std::lround(gx);
                int iy = (int)std::lround(gy);
                ix = std::clamp(ix, 0, mvW-1);
                iy = std::clamp(iy, 0, mvH-1);

                neigh.clear();
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        int nx = ix + ox, ny = iy + oy;
                        if (inb(nx, ny, mvW, mvH)) neigh.push_back(flowGrid[ny*mvW + nx]);
                    }
                }

                if (neigh.size() >= 5) {
                    F2 m = median_vec_2d(neigh);
                    float diff = std::hypot(f.x - m.x, f.y - m.y);
                    if (diff > neighborDiffThr) continue;
                }
            }

            sumX += (double)f.x;
            sumY += (double)f.y;
            count++;
        }
    }

    ResultantFlow rf{};
    rf.sum = F2{ (float)sumX, (float)sumY };
    rf.count = count;
    rf.avg = (count > 0) ? F2{ (float)(sumX / (double)count), (float)(sumY / (double)count) } : F2{0,0};
    return rf;
}

static void draw_resultant_arrow_on_frame(uint8_t* rgba, int W, int H,
                                         const F2& avgVec,
                                         float scalePxPerFlow, float maxLenPx,
                                         int thickness)
{
    int cx = W / 2;
    int cy = H / 2;

    float dx = avgVec.x * scalePxPerFlow;
    float dy = avgVec.y * scalePxPerFlow;

    float L = std::sqrt(dx*dx + dy*dy);
    if (L < 1e-4f) return;

    if (L > maxLenPx) {
        float s = maxLenPx / L;
        dx *= s; dy *= s;
    }

    int x1 = (int)std::lround((float)cx + dx);
    int y1 = (int)std::lround((float)cy + dy);

    draw_arrow_rgba_color(rgba, W, H, cx, cy, x1, y1, thickness, 255,0,0, 0,0,0);
}

// ---------------- GStreamer pipelines ----------------

static GstElement* build_input_pipeline_mp4_nvmm(const std::string &mp4Path, int W, int H)
{
    // NVDEC -> nvvidconv -> NVMM NV12 -> appsink
    std::string pipe =
        "filesrc location=\"" + mp4Path + "\" ! "
        "qtdemux name=dem "
        "dem.video_0 ! queue ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,framerate=30/1 ! "
        "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=false";

    GError* err = nullptr;
    GstElement* pipeline = gst_parse_launch(pipe.c_str(), &err);
    if (!pipeline) {
        std::cerr << "gst_parse_launch input failed: " << (err ? err->message : "unknown") << "\n";
        if (err) g_error_free(err);
        return nullptr;
    }
    if (err) g_error_free(err);
    return pipeline;
}



static GstElement* build_output_pipeline(const std::string &host, int port, int W, int H)
{
    // Kept identical to your original (software x264enc, appsrc RGBA).
    // Level B will switch to NVMM + nvv4l2h264enc for true end-to-end zero-copy.
    std::string pipe =
        "appsrc name=src is-live=true format=time do-timestamp=true "
        "caps=video/x-raw,format=RGBA,width=" + std::to_string(W) + ",height=" + std::to_string(H) +
        ",framerate=30/1,colorimetry=bt709,range=tv ! "
        "videoconvert ! "
        "video/x-raw,format=I420,width=" + std::to_string(W) + ",height=" + std::to_string(H) +
        ",framerate=30/1,colorimetry=bt709,range=tv ! "
        "x264enc tune=zerolatency speed-preset=superfast bitrate=30000 "
        "key-int-max=30 bframes=0 ref=1 sliced-threads=true threads=0 ! "
        "h264parse config-interval=-1 ! "
        "rtph264pay pt=96 config-interval=1 mtu=1200 ! "
        "udpsink host=" + host + " port=" + std::to_string(port) + " sync=false async=false";

    GError* err = nullptr;
    GstElement* pipeline = gst_parse_launch(pipe.c_str(), &err);
    if (!pipeline) {
        std::cerr << "gst_parse_launch output failed: " << (err ? err->message : "unknown") << "\n";
        if (err) g_error_free(err);
        return nullptr;
    }
    if (err) g_error_free(err);
    return pipeline;
}

static GstSample* pull_sample(GstAppSink* appsink, int timeout_ms)
{
    return gst_app_sink_try_pull_sample(appsink, (guint64)timeout_ms * 1000000ULL);
}

static bool push_rgba_frame(GstAppSrc* appsrc, const uint8_t* rgba, size_t bytes,
                            GstClockTime pts, GstClockTime dur)
{
    // NOTE: This does a CPU copy into a newly allocated GstBuffer (not zero-copy).
    GstBuffer* buf = gst_buffer_new_allocate(nullptr, bytes, nullptr);
    if (!buf) return false;

    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buf);
        return false;
    }
    std::memcpy(map.data, rgba, bytes);
    gst_buffer_unmap(buf, &map);

    GST_BUFFER_PTS(buf) = pts;
    GST_BUFFER_DURATION(buf) = dur;

    GstFlowReturn ret = gst_app_src_push_buffer(appsrc, buf);
    return ret == GST_FLOW_OK;
}

static int max_levels_scale_half_min32(int W, int H)
{
    int levels = 1;
    int w = W, h = H;
    while (true) {
        int w2 = (w + 1) / 2;
        int h2 = (h + 1) / 2;
        if (w2 < 32 || h2 < 32) break;
        w = w2; h = h2;
        levels++;
    }
    return levels;
}

// ---------------- NVMM -> DMABUF -> VPI wrapper (NVBUFFER) ----------------

static int get_dmabuf_fd_from_gstbuffer(GstBuffer *buf)
{
    guint n = gst_buffer_n_memory(buf);
    for (guint i = 0; i < n; ++i) {
        GstMemory *mem = gst_buffer_peek_memory(buf, i);
        if (mem && gst_is_dmabuf_memory(mem)) {
            return gst_dmabuf_memory_get_fd(mem);
        }
    }
    return -1;
}

static VPIImage wrap_dmabuf_fd_as_vpi_nvbuffer(int dmabuf_fd)
{
    // VPI 2.4.x VPIImageData has only bufferType + buffer union.
    // For NVBUFFER, you pass the NvBuffer fd directly in data.buffer.fd.
    VPIImageData data{};
    data.bufferType = VPI_IMAGE_BUFFER_NVBUFFER;
    data.buffer.fd  = dmabuf_fd;

    VPIImageWrapperParams params;
    CHECK_VPI(vpiInitImageWrapperParams(&params));

    VPIImage img = nullptr;
    CHECK_VPI(vpiImageCreateWrapper(&data, &params, 0, &img));
    return img;
}

// ---------------- MAIN ----------------

int main(int argc, char** argv)
{
    std::string mp4 = (argc >= 2) ? argv[1] : "/home/ntc-orin/Videos/StraightandBack_car_movement_always10kmph.mp4";
    int W = (argc >= 4) ? std::atoi(argv[2]) : 2560;
    int H = (argc >= 4) ? std::atoi(argv[3]) : 720;

    const float pyrScale  = 0.5f;
    const int   grid      = 2;
    const int   numLevels = max_levels_scale_half_min32(W, H);

    const int   STEP_BIG_PX   = 32;
    const float SCALE_BIG     = 14.0f;
    const float MAXLEN_BIG    = 80.0f;
    const float THRESH_MIN    = 0.05f;
    const int   THICKNESS_BIG = 3;

    const float SCALE_RESULTANT  = 120.0f;
    const float MAXLEN_RESULTANT = 160.0f;
    const int   THICKNESS_RESULTANT = 5;

    const bool  ENABLE_DYNAMIC_THR = true;
    const float DYN_P75_FACTOR     = 0.6f;

    const bool  ENABLE_NEIGH_GATE   = true;
    const float NEIGH_DIFF_THR      = 1.5f;

    std::cout
        << "LEVEL A: MP4(H264)->NVDEC->NVMM(DMABUF)->VPI(NVBUFFER wrapper)->OFA->CPU overlay->RTP\n"
        << "Input: " << mp4 << "\n"
        << "Size: " << W << "x" << H << "  Levels: " << numLevels << " grid=" << grid << "\n";

    gst_init(&argc, &argv);

    GstElement* inPipe  = build_input_pipeline_mp4_nvmm(mp4, W, H);
    GstElement* outPipe = build_output_pipeline("192.168.1.100", 5000, W, H);
    if (!inPipe || !outPipe) return 1;

    GstElement* sinkElem = gst_bin_get_by_name(GST_BIN(inPipe), "sink");
    GstElement* srcElem  = gst_bin_get_by_name(GST_BIN(outPipe), "src");
    if (!sinkElem || !srcElem) {
        std::cerr << "Could not get appsink/appsrc element\n";
        return 1;
    }

    GstAppSink* appsink = GST_APP_SINK(sinkElem);
    GstAppSrc*  appsrc  = GST_APP_SRC(srcElem);

    gst_element_set_state(outPipe, GST_STATE_PLAYING);
    gst_element_set_state(inPipe,  GST_STATE_PLAYING);

    GstBus* inBus  = gst_element_get_bus(inPipe);
    GstBus* outBus = gst_element_get_bus(outPipe);

    auto check_bus_errors = [&](GstBus* bus, const char* name)->bool {
        while (true) {
            GstMessage* msg = gst_bus_pop(bus);
            if (!msg) break;
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError* e = nullptr; gchar* dbg = nullptr;
                gst_message_parse_error(msg, &e, &dbg);
                std::cerr << name << " ERROR: " << (e ? e->message : "unknown") << "\n";
                if (dbg) std::cerr << "Debug: " << dbg << "\n";
                if (e) g_error_free(e);
                if (dbg) g_free(dbg);
                gst_message_unref(msg);
                return false;
            }
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
                gst_message_unref(msg);
                return false;
            }
            gst_message_unref(msg);
        }
        return true;
    };

    // ---- init VPI ----
    VPIStream stream = nullptr;
    CHECK_VPI(vpiStreamCreate(0, &stream));

    // We convert NV12 (wrapped) -> Y8 into these VPI-owned images
    VPIImage prev_y8_pl = nullptr, cur_y8_pl = nullptr;
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, 0, &prev_y8_pl));
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, 0, &cur_y8_pl));

    // Pyramids
    VPIPyramid prev_pyr_pl = nullptr, cur_pyr_pl = nullptr;
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, numLevels, pyrScale, 0, &prev_pyr_pl));
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, numLevels, pyrScale, 0, &cur_pyr_pl));

    VPIPyramid prev_pyr_bl = nullptr, cur_pyr_bl = nullptr;
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER_BL, numLevels, pyrScale, 0, &prev_pyr_bl));
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER_BL, numLevels, pyrScale, 0, &cur_pyr_bl));

    std::vector<VPIImage> prevLvlPL(numLevels, nullptr), curLvlPL(numLevels, nullptr);
    std::vector<VPIImage> prevLvlBL(numLevels, nullptr), curLvlBL(numLevels, nullptr);
    for (int lvl = 0; lvl < numLevels; ++lvl) {
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(prev_pyr_pl, lvl, &prevLvlPL[lvl]));
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(cur_pyr_pl,  lvl, &curLvlPL[lvl]));
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(prev_pyr_bl, lvl, &prevLvlBL[lvl]));
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(cur_pyr_bl,  lvl, &curLvlBL[lvl]));
    }

    std::vector<int32_t> gridArr(numLevels, grid);

    VPIPayload payload = nullptr;
    CHECK_VPI(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA,
                                        W, H,
                                        VPI_IMAGE_FORMAT_Y8_ER_BL,
                                        gridArr.data(), numLevels,
                                        VPI_OPTICAL_FLOW_QUALITY_HIGH,
                                        &payload));

    const int mvW = (W + grid - 1) / grid;
    const int mvH = (H + grid - 1) / grid;

    VPIImage mv_bl = nullptr, mv_pl = nullptr;
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16_BL, 0, &mv_bl));
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16,    0, &mv_pl));

    std::vector<uint8_t> prevRGBA;
    std::vector<uint8_t> rgba;
    std::vector<F2> flowGrid((size_t)mvW * (size_t)mvH);

    bool havePrev = false;

    const int fpsN = 30, fpsD = 1;
    GstClockTime frameDur = gst_util_uint64_scale_int(GST_SECOND, fpsD, fpsN);
    GstClockTime pts = 0;

    while (true) {
        if (!check_bus_errors(inBus, "IN") || !check_bus_errors(outBus, "OUT"))
            break;

        GstSample* sample = pull_sample(appsink, 2000);
        if (!sample) continue;

        GstBuffer* buf = gst_sample_get_buffer(sample);
        if (!buf) { gst_sample_unref(sample); continue; }

        int fd = get_dmabuf_fd_from_gstbuffer(buf);
        if (fd < 0) {
            std::cerr << "No DMABUF fd found. Appsink is not delivering NVMM/DMABUF.\n";
            gst_sample_unref(sample);
            continue;
        }

        // Wrap decoder output as VPI image (zero-copy view)
        VPIImage nv12Wrap = wrap_dmabuf_fd_as_vpi_nvbuffer(fd);

        // Convert NV12 -> Y8 into cur_y8_pl
        CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, nv12Wrap, cur_y8_pl, nullptr));
        CHECK_VPI(vpiStreamSync(stream));

        vpiImageDestroy(nv12Wrap);

        if (!havePrev) {
            // Build overlay base from current Y8 (readback via host pitch-linear lock)
            VPIImageData y8Data;
            CHECK_VPI(vpiImageLockData(cur_y8_pl, VPI_LOCK_READ,
                                       VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &y8Data));
            VPIImageLockGuard g(cur_y8_pl);

            auto &p = y8Data.buffer.pitch.planes[0];
            const uint8_t* yptr = (const uint8_t*)p.data;
            int ystride = (int)p.pitchBytes;

            gray_to_rgba_stride(prevRGBA, yptr, ystride, W, H);

            std::swap(prev_y8_pl, cur_y8_pl);
            havePrev = true;

            gst_sample_unref(sample);
            continue;
        }

        // Stage 1: Gaussian pyramids on CUDA
        CHECK_VPI(vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND_CUDA,
                                                   prev_y8_pl, prev_pyr_pl, VPI_BORDER_CLAMP));
        CHECK_VPI(vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND_CUDA,
                                                   cur_y8_pl,  cur_pyr_pl,  VPI_BORDER_CLAMP));
        CHECK_VPI(vpiStreamSync(stream));

        // Stage 2: PL -> BL conversion using VIC
        for (int lvl = 0; lvl < numLevels; ++lvl) {
            CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC,
                                                 prevLvlPL[lvl], prevLvlBL[lvl], nullptr));
            CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC,
                                                 curLvlPL[lvl],  curLvlBL[lvl],  nullptr));
        }
        CHECK_VPI(vpiStreamSync(stream));

        // Stage 3: OFA dense optical flow
        CHECK_VPI(vpiSubmitOpticalFlowDensePyramid(stream, VPI_BACKEND_OFA,
                                                   payload, prev_pyr_bl, cur_pyr_bl, mv_bl));
        CHECK_VPI(vpiStreamSync(stream));

        // Stage 4: MV BL -> PL for CPU readback
        CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, mv_bl, mv_pl, nullptr));
        CHECK_VPI(vpiStreamSync(stream));

        // Read MV to flowGrid
        {
            VPIImageData mvData;
            CHECK_VPI(vpiImageLockData(mv_pl, VPI_LOCK_READ,
                                       VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mvData));
            VPIImageLockGuard guard(mv_pl);

            auto &p0 = mvData.buffer.pitch.planes[0];
            const uint8_t* base = (const uint8_t*)p0.data;
            const int pitch = (int)p0.pitchBytes;

            for (int y = 0; y < mvH; ++y) {
                const int16_t* row = (const int16_t*)(base + (size_t)y * (size_t)pitch);
                for (int x = 0; x < mvW; ++x) {
                    float vx = s10_5_to_px(row[2*x + 0]);
                    float vy = s10_5_to_px(row[2*x + 1]);
                    flowGrid[(size_t)y * (size_t)mvW + (size_t)x] = F2{vx, vy};
                }
            }
        }

        ResultantFlow rf = compute_resultant_flow_no_texture(
            W, H,
            flowGrid.data(), mvW, mvH,
            grid,
            STEP_BIG_PX,
            THRESH_MIN,
            ENABLE_DYNAMIC_THR, DYN_P75_FACTOR,
            ENABLE_NEIGH_GATE, NEIGH_DIFF_THR
        );

        rgba = prevRGBA;

        draw_big_green_arrows_rgba_no_texture(
            rgba.data(), W, H,
            flowGrid.data(), mvW, mvH,
            grid,
            STEP_BIG_PX, 14.0f, 80.0f,
            THRESH_MIN, 3,
            ENABLE_DYNAMIC_THR, DYN_P75_FACTOR,
            ENABLE_NEIGH_GATE, NEIGH_DIFF_THR
        );

        draw_resultant_arrow_on_frame(rgba.data(), W, H, rf.avg,
                                      120.0f, 160.0f, 5);

        if (!push_rgba_frame(appsrc, rgba.data(), rgba.size(), pts, frameDur)) {
            std::cerr << "Failed to push buffer to appsrc\n";
            gst_sample_unref(sample);
            break;
        }
        pts += frameDur;

        // Update prevRGBA from current Y8 (readback)
        {
            VPIImageData y8Data;
            CHECK_VPI(vpiImageLockData(cur_y8_pl, VPI_LOCK_READ,
                                       VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &y8Data));
            VPIImageLockGuard g(cur_y8_pl);

            auto &p = y8Data.buffer.pitch.planes[0];
            const uint8_t* yptr = (const uint8_t*)p.data;
            int ystride = (int)p.pitchBytes;

            gray_to_rgba_stride(prevRGBA, yptr, ystride, W, H);
        }

        std::swap(prev_y8_pl, cur_y8_pl);

        gst_sample_unref(sample);
    }

    gst_app_src_end_of_stream(appsrc);
    CHECK_VPI(vpiStreamSync(stream));

    gst_element_set_state(inPipe,  GST_STATE_NULL);
    gst_element_set_state(outPipe, GST_STATE_NULL);

    if (inBus)  gst_object_unref(inBus);
    if (outBus) gst_object_unref(outBus);
    if (sinkElem) gst_object_unref(sinkElem);
    if (srcElem)  gst_object_unref(srcElem);
    if (inPipe)  gst_object_unref(inPipe);
    if (outPipe) gst_object_unref(outPipe);

    for (int lvl = 0; lvl < numLevels; ++lvl) {
        if (prevLvlPL[lvl]) vpiImageDestroy(prevLvlPL[lvl]);
        if (curLvlPL[lvl])  vpiImageDestroy(curLvlPL[lvl]);
        if (prevLvlBL[lvl]) vpiImageDestroy(prevLvlBL[lvl]);
        if (curLvlBL[lvl])  vpiImageDestroy(curLvlBL[lvl]);
    }
    if (mv_pl) vpiImageDestroy(mv_pl);
    if (mv_bl) vpiImageDestroy(mv_bl);
    if (payload) vpiPayloadDestroy(payload);
    if (prev_pyr_bl) vpiPyramidDestroy(prev_pyr_bl);
    if (cur_pyr_bl)  vpiPyramidDestroy(cur_pyr_bl);
    if (prev_pyr_pl) vpiPyramidDestroy(prev_pyr_pl);
    if (cur_pyr_pl)  vpiPyramidDestroy(cur_pyr_pl);
    if (prev_y8_pl) vpiImageDestroy(prev_y8_pl);
    if (cur_y8_pl)  vpiImageDestroy(cur_y8_pl);
    if (stream) vpiStreamDestroy(stream);

    return 0;
}
