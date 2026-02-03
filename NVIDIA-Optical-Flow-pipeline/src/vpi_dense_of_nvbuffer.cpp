// vpi_gst_of_rtp_overlay.cpp
//
// GStreamer PNG folder -> VPI OFA Dense Optical Flow (pyramids) -> draw arrows (no OpenCV)
// -> RTP/H264 UDP out
//
// Sender: 192.168.1.100:5000 (RTP/H264)
// Receiver example:
// gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
//   ! rtpjitterbuffer ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
//
// Notes:
// - This version adds "clean-up" logic to suppress random/outlier arrows:
//   1) Texture gate: don't draw arrows in flat/low-texture regions.
//   2) Local consistency gate: reject vectors that differ too much from neighbors (median).
//   3) Dynamic magnitude threshold: adapt threshold based on the current frame's flow magnitude distribution.
//
// Build: (example; adjust include/lib paths for your platform)
//   g++ -O2 -std=c++17 vpi_gst_of_rtp_overlay.cpp -o ofa_rtp \
//       `pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0` \
//       -lvpi
//

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

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstdint>

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

static bool file_exists(const std::string &p)
{
    FILE *f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// ---------------- Drawing (no OpenCV) ----------------

// Put a pixel in RGBA frame
static inline void put_px_rgba(uint8_t *img, int W, int H, int x, int y,
                               uint8_t r, uint8_t g, uint8_t b, uint8_t a=255)
{
    if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return;
    uint8_t *p = img + (size_t)(y * W + x) * 4;
    p[0] = r; p[1] = g; p[2] = b; p[3] = a;
}

// Bresenham line (thickness by simple square brush)
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

// Arrow: black outline + green inside (OpenCV-like style)
static void draw_arrow_rgba(uint8_t *img, int W, int H,
                            int x0, int y0, int x1, int y1,
                            int thickness)
{
    // main shaft outline
    draw_line_rgba(img, W, H, x0, y0, x1, y1, 0,0,0, thickness+2);
    draw_line_rgba(img, W, H, x0, y0, x1, y1, 0,255,0, thickness);

    // arrow head
    float dx = float(x1 - x0);
    float dy = float(y1 - y0);
    float L = std::sqrt(dx*dx + dy*dy);
    if (L < 1.0f) return;

    float ux = dx / L, uy = dy / L;
    float headLen = std::max(6.0f, float(thickness) * 3.0f);
    float ang = 0.6f; // ~34 deg
    float cx = float(x1), cy = float(y1);

    // rotate unit vector by +/- ang and go backwards
    float vx1 =  std::cos(ang)*ux - std::sin(ang)*uy;
    float vy1 =  std::sin(ang)*ux + std::cos(ang)*uy;
    float vx2 =  std::cos(-ang)*ux - std::sin(-ang)*uy;
    float vy2 =  std::sin(-ang)*ux + std::cos(-ang)*uy;

    int hx1 = int(std::lround(cx - headLen * vx1));
    int hy1 = int(std::lround(cy - headLen * vy1));
    int hx2 = int(std::lround(cx - headLen * vx2));
    int hy2 = int(std::lround(cy - headLen * vy2));

    draw_line_rgba(img, W, H, x1, y1, hx1, hy1, 0,0,0, thickness+2);
    draw_line_rgba(img, W, H, x1, y1, hx2, hy2, 0,0,0, thickness+2);
    draw_line_rgba(img, W, H, x1, y1, hx1, hy1, 0,255,0, thickness);
    draw_line_rgba(img, W, H, x1, y1, hx2, hy2, 0,255,0, thickness);
}

// Flow grid contains float2 per cell (vx,vy) in pixels.
struct F2 { float x,y; };

// Bilinear sample flow grid (mvW x mvH) at flow-grid coordinate gx,gy.
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

// ---------- Arrow clean-up helpers ----------

// Compute a cheap local texture score from GRAY8 image (sum of abs gradients).
// Higher means more texture; near 0 means flat region.
static inline float local_texture_4n(const uint8_t* g, int W, int H, int x, int y)
{
    int x1 = std::min(x+1, W-1), y1 = std::min(y+1, H-1);
    int x0 = std::max(x-1, 0),   y0 = std::max(y-1, 0);

    int gx = std::abs(int(g[y*W + x1]) - int(g[y*W + x0]));
    int gy = std::abs(int(g[y1*W + x]) - int(g[y0*W + x]));
    return float(gx + gy); // 0..510
}

static inline bool inb(int x,int y,int W,int H)
{
    return (unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H;
}

// Percentile (in-place nth_element). p01 in [0..1].
static float percentile_inplace(std::vector<float>& v, float p01)
{
    if (v.empty()) return 0.0f;
    p01 = std::clamp(p01, 0.0f, 1.0f);
    size_t k = (size_t)std::lround(p01 * float(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t)k, v.end());
    return v[k];
}

// Median of neighbor vectors (component-wise) using nth_element.
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

// Draw arrows with outlier rejection.
// - grayForTexture: optional pointer to GRAY8 (W*H) used for texture gating (can be nullptr to disable).
static void draw_big_green_arrows_rgba(
    uint8_t *rgba, int W, int H,
    const uint8_t* grayForTexture,
    const F2* flowGrid, int mvW, int mvH,
    int gridStep,             // OFA grid (e.g. 4)
    int stepPx,               // arrow sampling step in pixels (e.g. 64)
    float scale,              // arrow scale factor applied to flow
    float maxLenPx,           // arrow length clamp (pixels)
    float magThresholdMin,    // minimum magnitude threshold (pixels of flow)
    int thickness,

    // Clean-up knobs:
    bool  enableDynamicThreshold,
    float dynamicP75Factor,   // thr = max(minThr, p75 * factor)
    bool  enableTextureGate,
    float textureThreshold,   // 0..510, typical 10..30
    bool  enableNeighborGate,
    float neighborDiffThr     // in pixels of flow (before scaling), typical 1.2..2.5
)
{
    // ---- optional dynamic threshold pass (compute distribution of magnitudes) ----
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
        float dyn = p75 * dynamicP75Factor;
        magThr = std::max(magThresholdMin, dyn);
    }

    // ---- main drawing loop ----
    std::vector<F2> neigh; neigh.reserve(9);

    for (int y = 0; y < H; y += stepPx) {
        for (int x = 0; x < W; x += stepPx) {

            // 1) Texture gate: reject flat regions (optional)
            if (enableTextureGate && grayForTexture) {
                float tex = local_texture_4n(grayForTexture, W, H, x, y);
                if (tex < textureThreshold) continue;
            }

            // Sample flow at this location (bilinear in flow-grid coordinates)
            float gx = float(x) / float(gridStep);
            float gy = float(y) / float(gridStep);
            F2 f = bilinear_flow(flowGrid, mvW, mvH, gx, gy);

            float mag = std::sqrt(f.x*f.x + f.y*f.y);
            if (mag < magThr) continue;

            // 2) Neighbor consistency gate: reject isolated outliers (optional)
            if (enableNeighborGate) {
                int ix = (int)std::lround(gx);
                int iy = (int)std::lround(gy);
                ix = std::clamp(ix, 0, mvW-1);
                iy = std::clamp(iy, 0, mvH-1);

                neigh.clear();
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        int nx = ix + ox, ny = iy + oy;
                        if (inb(nx, ny, mvW, mvH)) {
                            neigh.push_back(flowGrid[ny*mvW + nx]);
                        }
                    }
                }

                if (neigh.size() >= 5) { // enough samples for a stable median
                    F2 m = median_vec_2d(neigh);
                    float diff = std::hypot(f.x - m.x, f.y - m.y);
                    if (diff > neighborDiffThr) continue;
                }
            }

            // Scale and clamp arrow length
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

// ---------------- VPI helpers ----------------

static void upload_gray8_to_vpi(VPIImage dstY8, const uint8_t* gray,
                               int W, int H, int srcStrideBytes)
{
    VPIImageData data;
    CHECK_VPI(vpiImageLockData(dstY8, VPI_LOCK_WRITE,
                              VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));
    VPIImageLockGuard guard(dstY8);

    auto &p = data.buffer.pitch.planes[0];
    for (int y = 0; y < H; ++y) {
        std::memcpy((uint8_t*)p.data + y * p.pitchBytes,
                    gray + y * (size_t)srcStrideBytes,
                    (size_t)W);
    }
}

// Copy possibly-strided GRAY8 into contiguous W*H
static void copy_to_contiguous_gray(std::vector<uint8_t> &dst, const uint8_t* src,
                                    int W, int H, int srcStrideBytes)
{
    dst.resize((size_t)W * (size_t)H);
    for (int y = 0; y < H; ++y) {
        std::memcpy(dst.data() + (size_t)y * (size_t)W,
                    src + (size_t)y * (size_t)srcStrideBytes,
                    (size_t)W);
    }
}

// Create RGBA from GRAY8
static void gray_to_rgba(std::vector<uint8_t> &rgba, const std::vector<uint8_t> &gray, int W, int H)
{
    rgba.resize((size_t)W * (size_t)H * 4);
    const uint8_t *g = gray.data();
    uint8_t *p = rgba.data();
    for (size_t i = 0; i < (size_t)W*(size_t)H; ++i) {
        uint8_t v = g[i];
        p[4*i+0] = v;
        p[4*i+1] = v;
        p[4*i+2] = v;
        p[4*i+3] = 255;
    }
}

// ---------------- GStreamer pipelines ----------------

static GstElement* build_input_pipeline(const std::string &dir, int W, int H)
{
    // multifilesrc -> pngdec -> videoconvert -> videoscale -> GRAY8 -> appsink
    std::string pipe =
        "multifilesrc location=" + dir + "/frame_%06d.png start-index=0 caps=image/png,framerate=30/1 ! "
        "pngdec ! videoconvert ! videoscale ! "
        "video/x-raw,format=GRAY8,width=" + std::to_string(W) + ",height=" + std::to_string(H) + ",framerate=30/1 ! "
        "appsink name=sink emit-signals=false sync=false max-buffers=2 drop=true";

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
    // appsrc (RGBA) -> videoconvert -> I420 -> x264enc(superfast) -> RTP/H264 -> UDP
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

    GstFlowReturn ret = gst_app_src_push_buffer(appsrc, buf); // ownership transferred
    return ret == GST_FLOW_OK;
}

int main(int argc, char** argv)
{
    std::string dir = (argc >= 2) ? argv[1] : "/path/to/images";
    int W = (argc >= 4) ? std::atoi(argv[2]) : 2560;
    int H = (argc >= 4) ? std::atoi(argv[3]) : 720;

    // Quick sanity check for first file (helps avoid confusing multifilesrc errors)
    std::string first = dir + "/frame_000000.png";
    if (!file_exists(first)) {
        std::cerr << "Missing first frame: " << first << "\n";
        return 1;
    }

    // ---- OFA/VPI knobs ----
    const int   numLevels = 4;
    const float pyrScale  = 0.5f;
    const int   grid      = 4;

    // Overlay knobs
    const int   STEP_BIG_PX   = 64;
    const float SCALE_BIG     = 14.0f;
    const float MAXLEN_BIG    = 80.0f;
    const float THRESH_MIN    = 0.05f; // minimum flow magnitude (pixels) before drawing
    const int   THICKNESS_BIG = 3;

    // ---- Clean-up knobs (tune these to remove "random arrows") ----
    const bool  ENABLE_DYNAMIC_THR = true;
    const float DYN_P75_FACTOR     = 0.6f;  // thr = max(THRESH_MIN, p75 * factor). Increase to show fewer arrows.

    const bool  ENABLE_TEXTURE_GATE = true;
    const float TEXTURE_THR         = 15.0f; // 0..510. Increase to suppress arrows in flat regions.

    const bool  ENABLE_NEIGH_GATE   = true;
    const float NEIGH_DIFF_THR      = 1.5f;  // flow pixels (before scaling). Lower = stricter outlier rejection.

    std::cout << "PNG(folder) -> VPI OFA Dense OF -> overlay arrows -> RTP/H264 UDP\n"
              << "Dir: " << dir << "\n"
              << "Size: " << W << "x" << H << "\n"
              << "Levels: " << numLevels << " scale=" << pyrScale << " grid=" << grid << "\n"
              << "Send: 192.168.1.100:5000 (RTP/H264)\n"
              << "Cleanup: dynThr=" << ENABLE_DYNAMIC_THR
              << " texGate=" << ENABLE_TEXTURE_GATE
              << " neighGate=" << ENABLE_NEIGH_GATE << "\n";

    // ---- init GStreamer ----
    gst_init(&argc, &argv);

    GstElement* inPipe  = build_input_pipeline(dir, W, H);
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

    // Start both pipelines
    gst_element_set_state(outPipe, GST_STATE_PLAYING);
    gst_element_set_state(inPipe,  GST_STATE_PLAYING);

    GstBus* inBus  = gst_element_get_bus(inPipe);
    GstBus* outBus = gst_element_get_bus(outPipe);

    // ---- init VPI ----
    VPIStream stream = nullptr;
    CHECK_VPI(vpiStreamCreate(0, &stream));

    VPIImage prev_y8_pl = nullptr, cur_y8_pl = nullptr;
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, 0, &prev_y8_pl));
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, 0, &cur_y8_pl));

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

    // Host buffers
    std::vector<uint8_t> prevGray, curGray, rgba;
    std::vector<F2> flowGrid((size_t)mvW * (size_t)mvH);

    bool havePrev = false;

    // Timing for RTP (30 fps from pipeline caps); we push one per input sample
    const int fpsN = 30, fpsD = 1;
    GstClockTime frameDur = gst_util_uint64_scale_int(GST_SECOND, fpsD, fpsN);
    GstClockTime pts = 0;

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
                std::cout << name << " EOS\n";
                gst_message_unref(msg);
                return false;
            }
            gst_message_unref(msg);
        }
        return true;
    };

    while (true) {
        if (!check_bus_errors(inBus, "IN") || !check_bus_errors(outBus, "OUT"))
            break;

        GstSample* sample = pull_sample(appsink, 2000);
        if (!sample) continue;

        GstBuffer* buf = gst_sample_get_buffer(sample);
        if (!buf) { gst_sample_unref(sample); continue; }

        GstMapInfo map;
        if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            continue;
        }

        int stride = W;
        if (GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf)) {
            stride = (int)vmeta->stride[0];
        }
        const uint8_t* gray = (const uint8_t*)map.data;

        copy_to_contiguous_gray(curGray, gray, W, H, stride);

        gst_buffer_unmap(buf, &map);
        gst_sample_unref(sample);

        if (!havePrev) {
            prevGray = curGray;
            havePrev = true;
            continue;
        }

        // ---- VPI staged sync ----

        upload_gray8_to_vpi(prev_y8_pl, prevGray.data(), W, H, W);
        upload_gray8_to_vpi(cur_y8_pl,  curGray.data(),  W, H, W);

        // Stage 1: pyramids (CPU)
        CHECK_VPI(vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND_CPU,
                                                   prev_y8_pl, prev_pyr_pl, VPI_BORDER_CLAMP));
        CHECK_VPI(vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND_CPU,
                                                   cur_y8_pl,  cur_pyr_pl,  VPI_BORDER_CLAMP));
        CHECK_VPI(vpiStreamSync(stream));

        // Stage 2: PL -> BL per level (VIC)
        for (int lvl = 0; lvl < numLevels; ++lvl) {
            CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC,
                                                 prevLvlPL[lvl], prevLvlBL[lvl], nullptr));
            CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC,
                                                 curLvlPL[lvl],  curLvlBL[lvl],  nullptr));
        }
        CHECK_VPI(vpiStreamSync(stream));

        // Stage 3: OFA
        CHECK_VPI(vpiSubmitOpticalFlowDensePyramid(stream, VPI_BACKEND_OFA,
                                                   payload, prev_pyr_bl, cur_pyr_bl, mv_bl));
        CHECK_VPI(vpiStreamSync(stream));

        // Stage 4: mv BL -> mv PL
        CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, mv_bl, mv_pl, nullptr));
        CHECK_VPI(vpiStreamSync(stream));

        // Read mv_pl -> flowGrid float (pixels)
        {
            VPIImageData mvData;
            CHECK_VPI(vpiImageLockData(mv_pl, VPI_LOCK_READ,
                                       VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mvData));
            VPIImageLockGuard guard(mv_pl);

            auto &p0 = mvData.buffer.pitch.planes[0];
            const uint8_t* base = (const uint8_t*)p0.data;
            const int pitch = p0.pitchBytes;

            for (int y = 0; y < mvH; ++y) {
                const int16_t* row = (const int16_t*)(base + (size_t)y * (size_t)pitch);
                for (int x = 0; x < mvW; ++x) {
                    float vx = s10_5_to_px(row[2*x + 0]);
                    float vy = s10_5_to_px(row[2*x + 1]);
                    flowGrid[(size_t)y * (size_t)mvW + (size_t)x] = F2{vx, vy};
                }
            }
        }

        // Build RGBA frame from prevGray (overlay is drawn on previous frame)
        gray_to_rgba(rgba, prevGray, W, H);

        // Draw arrows with clean-up logic
        draw_big_green_arrows_rgba(
            rgba.data(), W, H,
            prevGray.data(),
            flowGrid.data(), mvW, mvH,
            grid,
            STEP_BIG_PX, SCALE_BIG, MAXLEN_BIG,
            THRESH_MIN, THICKNESS_BIG,
            ENABLE_DYNAMIC_THR, DYN_P75_FACTOR,
            ENABLE_TEXTURE_GATE, TEXTURE_THR,
            ENABLE_NEIGH_GATE, NEIGH_DIFF_THR
        );

        // Push out RTP/H264 frame
        if (!push_rgba_frame(appsrc, rgba.data(), rgba.size(), pts, frameDur)) {
            std::cerr << "Failed to push buffer to appsrc\n";
            break;
        }
        pts += frameDur;

        // Advance frames
        prevGray.swap(curGray);
    }

    // Tell appsrc end-of-stream
    gst_app_src_end_of_stream(appsrc);

    CHECK_VPI(vpiStreamSync(stream));

    // Cleanup GStreamer
    gst_element_set_state(inPipe,  GST_STATE_NULL);
    gst_element_set_state(outPipe, GST_STATE_NULL);

    if (inBus)  gst_object_unref(inBus);
    if (outBus) gst_object_unref(outBus);

    if (sinkElem) gst_object_unref(sinkElem);
    if (srcElem)  gst_object_unref(srcElem);

    if (inPipe)  gst_object_unref(inPipe);
    if (outPipe) gst_object_unref(outPipe);

    // Cleanup VPI
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
